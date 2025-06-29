import matter from 'gray-matter';
import { format, parseISO } from 'date-fns';

/**
 * Blog system configuration and utilities
 * Handles loading and parsing of markdown blog posts from static files
 */

// Blog configuration constants
export const BLOG_CONFIG = {
  categories: {
    curiosities: {
      name: 'Mathematical Curiosities',
      description: 'Explorations of games, puzzles, and mathematical phenomena',
      color: 'blue'
    },
    research: {
      name: 'Research Notes',
      description: 'Academic papers, studies, and research insights',
      color: 'indigo'
    }
  },
  postsPerPage: 6
};

// Cache configuration for performance optimization
let postsCache = null;
let lastCacheTime = null;
const CACHE_DURATION = 5 * 60 * 1000; // 5 minutes

/**
 * Constructs the correct asset path for GitHub Pages deployment
 * @param {string} path - Relative path to the asset
 * @returns {string} - Complete path including PUBLIC_URL if available
 */
function getAssetPath(path) {
  const cleanPath = path.startsWith('/') ? path.substring(1) : path;
  return process.env.PUBLIC_URL ? `${process.env.PUBLIC_URL}/${cleanPath}` : `/${cleanPath}`;
}

/**
 * Load and parse a markdown file from the public directory
 * @param {string} filePath - Path to the markdown file relative to public/
 * @returns {Object|null} - Parsed markdown with frontmatter and content, or null if failed
 */
export async function loadMarkdownFile(filePath) {
  try {
    const fullPath = getAssetPath(filePath);
    console.log(`🔍 Loading markdown file: ${fullPath}`);
    
    const response = await fetch(fullPath);
    
    if (!response.ok) {
      throw new Error(`Failed to load ${fullPath}: ${response.status}`);
    }
    
    const content = await response.text();
    const parsed = matter(content);
    
    console.log(`✅ Successfully loaded: ${filePath}`);
    return parsed;
  } catch (error) {
    console.error(`❌ Error loading markdown file ${filePath}:`, error);
    return null;
  }
}

/**
 * Get the list of all available posts from the manifest file
 * @returns {Array} - Array of post metadata objects
 */
export async function getPostsManifest() {
  try {
    const manifestPath = getAssetPath('blog/posts-manifest.json');
    console.log(`🔍 Loading posts manifest from: ${manifestPath}`);
    
    const response = await fetch(manifestPath);
    
    if (!response.ok) {
      console.warn('⚠️ Posts manifest not found, using empty list');
      return [];
    }
    
    const manifest = await response.json();
    console.log(`✅ Loaded manifest with ${manifest.length} posts`);
    
    return manifest;
  } catch (error) {
    console.error('❌ Error loading posts manifest:', error);
    return [];
  }
}

/**
 * Load and parse all blog posts with caching
 * @returns {Array} - Array of complete post objects sorted by date (newest first)
 */
export async function loadAllPosts() {
  console.log('🚀 Starting to load all posts...');
  
  // Return cached posts if still valid
  if (postsCache && lastCacheTime && Date.now() - lastCacheTime < CACHE_DURATION) {
    console.log('📦 Returning cached posts');
    return postsCache;
  }

  try {
    const manifest = await getPostsManifest();
    
    if (manifest.length === 0) {
      console.warn('⚠️ No posts found in manifest');
      return [];
    }
    
    const posts = [];

    // Process each post in the manifest
    for (const postInfo of manifest) {
      console.log(`📖 Processing post: ${postInfo.filename} from ${postInfo.category}`);
      
      const postPath = `blog/posts/${postInfo.category}/${postInfo.filename}`;
      const postData = await loadMarkdownFile(postPath);
      
      if (postData) {
        const post = {
          ...postData.data,
          content: postData.content,
          slug: postInfo.filename.replace('.md', ''),
          category: postInfo.category,
          readingTime: calculateReadingTime(postData.content),
          excerpt: postData.data.excerpt || generateExcerpt(postData.content)
        };

        // Validate required fields before adding to posts array
        if (post.title && post.date) {
          posts.push(post);
          console.log(`✅ Added post: ${post.title}`);
        } else {
          console.warn(`⚠️ Post ${postInfo.filename} missing required fields (title: ${!!post.title}, date: ${!!post.date})`);
        }
      } else {
        console.error(`❌ Failed to load post data for ${postInfo.filename}`);
      }
    }

    console.log(`🎉 Successfully loaded ${posts.length} posts`);

    // Sort posts by date (newest first)
    posts.sort((a, b) => new Date(b.date) - new Date(a.date));

    // Cache the results
    postsCache = posts;
    lastCacheTime = Date.now();

    return posts;
  } catch (error) {
    console.error('❌ Error loading posts:', error);
    return [];
  }
}

/**
 * Get a single post by category and slug
 * @param {string} category - Post category
 * @param {string} slug - Post slug (filename without .md extension)
 * @returns {Object|null} - Complete post object or null if not found
 */
export async function getPostBySlug(category, slug) {
  try {
    console.log(`🔍 Loading post: ${category}/${slug}`);
    
    const postPath = `blog/posts/${category}/${slug}.md`;
    const postData = await loadMarkdownFile(postPath);
    
    if (!postData) {
      console.warn(`⚠️ Post not found: ${category}/${slug}`);
      return null;
    }

    const post = {
      ...postData.data,
      content: postData.content,
      slug,
      category,
      readingTime: calculateReadingTime(postData.content),
      excerpt: postData.data.excerpt || generateExcerpt(postData.content)
    };

    console.log(`✅ Successfully loaded post: ${post.title}`);
    return post;
  } catch (error) {
    console.error(`❌ Error loading post ${category}/${slug}:`, error);
    return null;
  }
}

/**
 * Get all posts filtered by category
 * @param {string} category - Category to filter by
 * @returns {Array} - Array of posts in the specified category
 */
export async function getPostsByCategory(category) {
  console.log(`🔍 Loading posts for category: ${category}`);
  const allPosts = await loadAllPosts();
  const filtered = allPosts.filter(post => post.category === category);
  console.log(`✅ Found ${filtered.length} posts in category: ${category}`);
  return filtered;
}

/**
 * Get all posts filtered by tag
 * @param {string} tag - Tag to filter by
 * @returns {Array} - Array of posts with the specified tag
 */
export async function getPostsByTag(tag) {
  console.log(`🔍 Loading posts for tag: ${tag}`);
  const allPosts = await loadAllPosts();
  const filtered = allPosts.filter(post => 
    post.tags && post.tags.some(postTag => 
      postTag.toLowerCase() === tag.toLowerCase()
    )
  );
  console.log(`✅ Found ${filtered.length} posts with tag: ${tag}`);
  return filtered;
}

/**
 * Get all unique tags from all posts
 * @returns {Array} - Sorted array of unique tags
 */
export async function getAllTags() {
  const allPosts = await loadAllPosts();
  const tagSet = new Set();
  
  allPosts.forEach(post => {
    if (post.tags) {
      post.tags.forEach(tag => tagSet.add(tag));
    }
  });
  
  const tags = Array.from(tagSet).sort();
  console.log(`📊 Found ${tags.length} unique tags`);
  return tags;
}

/**
 * Calculate estimated reading time for content
 * @param {string} content - Markdown content
 * @returns {number} - Estimated reading time in minutes
 */
function calculateReadingTime(content) {
  const wordsPerMinute = 200;
  const wordCount = content.trim().split(/\s+/).length;
  const readingTime = Math.ceil(wordCount / wordsPerMinute);
  return readingTime;
}

/**
 * Generate excerpt from markdown content
 * @param {string} content - Markdown content
 * @param {number} maxLength - Maximum length of excerpt
 * @returns {string} - Generated excerpt
 */
function generateExcerpt(content, maxLength = 160) {
  // Remove markdown syntax for excerpt
  const plainText = content
    .replace(/#{1,6}\s+/g, '') // Headers
    .replace(/\*\*(.*?)\*\*/g, '$1') // Bold
    .replace(/\*(.*?)\*/g, '$1') // Italic
    .replace(/`(.*?)`/g, '$1') // Inline code
    .replace(/\[(.*?)\]\(.*?\)/g, '$1') // Links
    .replace(/!\[.*?\]\(.*?\)/g, '') // Images
    .replace(/\$\$(.*?)\$\$/g, '[Math]') // Display math
    .replace(/\$(.*?)\$/g, '[Math]') // Inline math
    .trim();

  if (plainText.length <= maxLength) {
    return plainText;
  }

  return plainText.substring(0, maxLength).replace(/\s+\S*$/, '') + '...';
}

/**
 * Format date for display using date-fns
 * @param {string|Date} dateString - Date string or Date object
 * @param {string} formatStr - Format string for date-fns
 * @returns {string} - Formatted date string
 */
export function formatDate(dateString, formatStr = 'MMMM d, yyyy') {
  try {
    const date = typeof dateString === 'string' ? parseISO(dateString) : dateString;
    return format(date, formatStr);
  } catch (error) {
    console.error('❌ Error formatting date:', error);
    return dateString;
  }
}

/**
 * Generate URL-friendly slug from text
 * @param {string} text - Text to convert to slug
 * @returns {string} - URL-friendly slug
 */
export function slugify(text) {
  return text
    .toString()
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .toLowerCase()
    .trim()
    .replace(/\s+/g, '-')
    .replace(/[^\w\-]+/g, '')
    .replace(/\-\-+/g, '-');
}

/**
 * Clear the posts cache (useful for development and testing)
 */
export function clearPostsCache() {
  console.log('🗑️ Clearing posts cache');
  postsCache = null;
  lastCacheTime = null;
}