import matter from 'gray-matter';
import { format, parseISO } from 'date-fns';

// Blog configuration
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

// Cache for loaded posts
let postsCache = null;
let lastCacheTime = null;
const CACHE_DURATION = 5 * 60 * 1000; // 5 minutes

/**
 * Load a markdown file and parse its content
 */
export async function loadMarkdownFile(filePath) {
  try {
    const response = await fetch(filePath);
    if (!response.ok) {
      throw new Error(`Failed to load ${filePath}: ${response.status}`);
    }
    const content = await response.text();
    return matter(content);
  } catch (error) {
    console.error(`Error loading markdown file ${filePath}:`, error);
    return null;
  }
}

/**
 * Get list of all available posts from the manifest
 */
export async function getPostsManifest() {
  try {
    const response = await fetch('/blog/posts-manifest.json');
    if (!response.ok) {
      console.warn('Posts manifest not found, using empty list');
      return [];
    }
    return await response.json();
  } catch (error) {
    console.error('Error loading posts manifest:', error);
    return [];
  }
}

/**
 * Load and parse all blog posts
 */
export async function loadAllPosts() {
  // Return cached posts if still valid
  if (postsCache && lastCacheTime && Date.now() - lastCacheTime < CACHE_DURATION) {
    return postsCache;
  }

  try {
    const manifest = await getPostsManifest();
    const posts = [];

    for (const postInfo of manifest) {
      const postData = await loadMarkdownFile(`/blog/posts/${postInfo.category}/${postInfo.filename}`);
      
      if (postData) {
        const post = {
          ...postData.data,
          content: postData.content,
          slug: postInfo.filename.replace('.md', ''),
          category: postInfo.category,
          readingTime: calculateReadingTime(postData.content),
          excerpt: postData.data.excerpt || generateExcerpt(postData.content)
        };

        // Validate required fields
        if (post.title && post.date) {
          posts.push(post);
        } else {
          console.warn(`Post ${postInfo.filename} missing required fields (title, date)`);
        }
      }
    }

    // Sort posts by date (newest first)
    posts.sort((a, b) => new Date(b.date) - new Date(a.date));

    // Cache the results
    postsCache = posts;
    lastCacheTime = Date.now();

    return posts;
  } catch (error) {
    console.error('Error loading posts:', error);
    return [];
  }
}

/**
 * Get a single post by slug and category
 */
export async function getPostBySlug(category, slug) {
  try {
    const postData = await loadMarkdownFile(`/blog/posts/${category}/${slug}.md`);
    
    if (!postData) {
      return null;
    }

    return {
      ...postData.data,
      content: postData.content,
      slug,
      category,
      readingTime: calculateReadingTime(postData.content),
      excerpt: postData.data.excerpt || generateExcerpt(postData.content)
    };
  } catch (error) {
    console.error(`Error loading post ${category}/${slug}:`, error);
    return null;
  }
}

/**
 * Get posts filtered by category
 */
export async function getPostsByCategory(category) {
  const allPosts = await loadAllPosts();
  return allPosts.filter(post => post.category === category);
}

/**
 * Get posts filtered by tag
 */
export async function getPostsByTag(tag) {
  const allPosts = await loadAllPosts();
  return allPosts.filter(post => 
    post.tags && post.tags.some(postTag => 
      postTag.toLowerCase() === tag.toLowerCase()
    )
  );
}

/**
 * Get all unique tags from all posts
 */
export async function getAllTags() {
  const allPosts = await loadAllPosts();
  const tagSet = new Set();
  
  allPosts.forEach(post => {
    if (post.tags) {
      post.tags.forEach(tag => tagSet.add(tag));
    }
  });
  
  return Array.from(tagSet).sort();
}

/**
 * Calculate estimated reading time
 */
function calculateReadingTime(content) {
  const wordsPerMinute = 200;
  const wordCount = content.trim().split(/\s+/).length;
  const readingTime = Math.ceil(wordCount / wordsPerMinute);
  return readingTime;
}

/**
 * Generate excerpt from content
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
 * Format date for display
 */
export function formatDate(dateString, formatStr = 'MMMM d, yyyy') {
  try {
    const date = typeof dateString === 'string' ? parseISO(dateString) : dateString;
    return format(date, formatStr);
  } catch (error) {
    console.error('Error formatting date:', error);
    return dateString;
  }
}

/**
 * Generate URL-friendly slug from title
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
 * Clear posts cache (useful for development)
 */
export function clearPostsCache() {
  postsCache = null;
  lastCacheTime = null;
}