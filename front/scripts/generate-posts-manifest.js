#!/usr/bin/env node

const fs = require('fs');
const path = require('path');

// Configuration
const BLOG_DIR = path.join(__dirname, '..', 'public', 'blog', 'posts');
const MANIFEST_PATH = path.join(__dirname, '..', 'public', 'blog', 'posts-manifest.json');

/**
 * Recursively scan directory for markdown files
 */
function scanDirectory(dir, basePath = '') {
  const items = [];
  
  try {
    const entries = fs.readdirSync(dir, { withFileTypes: true });
    
    for (const entry of entries) {
      const fullPath = path.join(dir, entry.name);
      
      if (entry.isDirectory()) {
        // Recursively scan subdirectories (categories)
        const categoryPath = path.join(basePath, entry.name);
        items.push(...scanDirectory(fullPath, categoryPath));
      } else if (entry.isFile() && entry.name.endsWith('.md')) {
        // Found a markdown file
        const category = basePath || 'uncategorized';
        items.push({
          filename: entry.name,
          category: category,
          path: path.join(basePath, entry.name),
          lastModified: fs.statSync(fullPath).mtime.toISOString()
        });
      }
    }
  } catch (error) {
    console.error(`Error scanning directory ${dir}:`, error.message);
  }
  
  return items;
}

/**
 * Generate posts manifest
 */
function generateManifest() {
  console.log('Generating posts manifest...');
  
  // Check if blog posts directory exists
  if (!fs.existsSync(BLOG_DIR)) {
    console.error(`Blog posts directory not found: ${BLOG_DIR}`);
    console.log('Creating directory structure...');
    
    // Create directory structure
    fs.mkdirSync(BLOG_DIR, { recursive: true });
    fs.mkdirSync(path.join(BLOG_DIR, 'curiosities'), { recursive: true });
    fs.mkdirSync(path.join(BLOG_DIR, 'research'), { recursive: true });
    
    console.log('Created blog directory structure. Please add your markdown posts.');
    return;
  }
  
  // Scan for posts
  const posts = scanDirectory(BLOG_DIR);
  
  if (posts.length === 0) {
    console.log('No markdown posts found. Please add posts to the blog/posts directory.');
    // Create empty manifest
    fs.writeFileSync(MANIFEST_PATH, JSON.stringify([], null, 2));
    return;
  }
  
  // Sort posts by category and filename
  posts.sort((a, b) => {
    if (a.category !== b.category) {
      return a.category.localeCompare(b.category);
    }
    return a.filename.localeCompare(b.filename);
  });
  
  // Create simplified manifest (without full paths and timestamps for client)
  const manifest = posts.map(post => ({
    filename: post.filename,
    category: post.category
  }));
  
  // Write manifest file
  try {
    fs.writeFileSync(MANIFEST_PATH, JSON.stringify(manifest, null, 2));
    console.log(`✅ Generated manifest with ${posts.length} posts:`);
    
    // Group by category for display
    const byCategory = {};
    posts.forEach(post => {
      if (!byCategory[post.category]) {
        byCategory[post.category] = [];
      }
      byCategory[post.category].push(post.filename);
    });
    
    Object.entries(byCategory).forEach(([category, files]) => {
      console.log(`  📁 ${category}: ${files.length} posts`);
      files.forEach(file => {
        console.log(`    📄 ${file}`);
      });
    });
    
  } catch (error) {
    console.error('Error writing manifest file:', error.message);
    process.exit(1);
  }
}

/**
 * Watch for changes and regenerate manifest
 */
function watchForChanges() {
  console.log('\n👀 Watching for changes in blog posts...');
  console.log('Press Ctrl+C to stop watching.\n');
  
  try {
    fs.watch(BLOG_DIR, { recursive: true }, (eventType, filename) => {
      if (filename && filename.endsWith('.md')) {
        console.log(`📝 Detected change: ${filename}`);
        setTimeout(generateManifest, 100); // Debounce
      }
    });
  } catch (error) {
    console.error('Error setting up file watcher:', error.message);
  }
}

// Main execution
function main() {
  const args = process.argv.slice(2);
  const shouldWatch = args.includes('--watch') || args.includes('-w');
  
  console.log('📚 Blog Posts Manifest Generator');
  console.log('================================\n');
  
  generateManifest();
  
  if (shouldWatch) {
    watchForChanges();
  } else {
    console.log('\n💡 Tip: Use --watch to automatically regenerate when posts change');
    console.log('   npm run generate-manifest -- --watch');
  }
}

if (require.main === module) {
  main();
}

module.exports = { generateManifest, scanDirectory };