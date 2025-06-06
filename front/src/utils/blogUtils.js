// Updated blogUtils.js with better debugging
import matter from 'gray-matter';
import { format, parseISO } from 'date-fns';

// Base path for assets when deployed to GitHub Pages
const base = process.env.PUBLIC_URL || '';

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
  console.log(`🔍 Attempting to load: ${filePath}`);
  
  try {
    const response = await fetch(filePath);
    console.log(`📡 Response status for ${filePath}: ${response.status}`);
    
    if (!response.ok) {
      throw new Error(`Failed to load ${filePath}: ${response.status}`);
    }
    const content = await response.text();
    console.log(`✅ Successfully loaded ${filePath}, content length: ${content.length}`);
    
    const parsed = matter(content);
    console.log(`📋 Parsed frontmatter:`, parsed.data);
    
    return parsed;
  } catch (error) {
    console.error(`❌ Error loading markdown file ${filePath}:`, error);
    return null;
  }
}

/**
 * Get list of all available posts from the manifest
 */
export async function getPostsManifest() {
  console.log('🔍 Loading posts manifest...');
  
  try {
    const response = await fetch(`${base}/blog/posts-manifest.json`);
    console.log(`📡 Manifest response status: ${response.status}`);
    
    if (!response.ok) {
      console.warn('⚠️ Posts manifest not found, using empty list');
      return [];
    }
    
    const manifest = await response.json();
    console.log(`✅ Loaded manifest with ${manifest.length} posts:`, manifest);
    
    return manifest;
  } catch (error) {
    console.error('❌ Error loading posts manifest:', error);
    return [];
  }
}

/**
 * Load and parse all blog posts
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
      console.warn('⚠️ No posts in manifest');
      return [];
    }
    
    const posts = [];

    for (const postInfo of manifest) {
      const fullPath = `${base}/blog/posts/${postInfo.category}/${postInfo.filename}`;
      console.log(`📖 Processing post: ${postInfo.filename} from ${postInfo.category}`);
      
      const postData = await loadMarkdownFile(fullPath);
      
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
          console.log(`✅ Added post: ${post.title}`);
        } else {
          console.warn(`⚠️ Post ${postInfo.filename} missing required fields (title: ${post.title}, date: ${post.date})`);
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

// ... rest of the functions remain the same
export async function getPostBySlug(category, slug) {
  try {
    const postData = await loadMarkdownFile(`${base}/blog/posts/${category}/${slug}.md`);
    
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

export async function getPostsByCategory(category) {
  const allPosts = await loadAllPosts();
  return allPosts.filter(post => post.category === category);
}

export async function getPostsByTag(tag) {
  const allPosts = await loadAllPosts();
  return allPosts.filter(post => 
    post.tags && post.tags.some(postTag => 
      postTag.toLowerCase() === tag.toLowerCase()
    )
  );
}

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

function calculateReadingTime(content) {
  const wordsPerMinute = 200;
  const wordCount = content.trim().split(/\s+/).length;
  const readingTime = Math.ceil(wordCount / wordsPerMinute);
  return readingTime;
}

function generateExcerpt(content, maxLength = 160) {
  const plainText = content
    .replace(/#{1,6}\s+/g, '')
    .replace(/\*\*(.*?)\*\*/g, '$1')
    .replace(/\*(.*?)\*/g, '$1')
    .replace(/`(.*?)`/g, '$1')
    .replace(/\[(.*?)\]\(.*?\)/g, '$1')
    .replace(/!\[.*?\]\(.*?\)/g, '')
    .replace(/\$\$(.*?)\$\$/g, '[Math]')
    .replace(/\$(.*?)\$/g, '[Math]')
    .trim();

  if (plainText.length <= maxLength) {
    return plainText;
  }

  return plainText.substring(0, maxLength).replace(/\s+\S*$/, '') + '...';
}

export function formatDate(dateString, formatStr = 'MMMM d, yyyy') {
  try {
    const date = typeof dateString === 'string' ? parseISO(dateString) : dateString;
    return format(date, formatStr);
  } catch (error) {
    console.error('Error formatting date:', error);
    return dateString;
  }
}

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

export function clearPostsCache() {
  postsCache = null;
  lastCacheTime = null;
}