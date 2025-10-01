import { useState, useEffect, useRef } from 'react';
import { Link } from 'react-router-dom';
import { Helmet } from 'react-helmet-async';
import { motion, useScroll, useTransform } from 'framer-motion';
import { 
  Calendar, 
  Clock, 
  Tag, 
  Search, 
  Filter, 
  BookOpen, 
  ArrowRight, 
  Layers,
  Brain,
  FileText,
  Mail
} from 'lucide-react';
import { loadAllPosts, getAllTags, BLOG_CONFIG, formatDate } from '../utils/blogUtils';
import { variants as motionVariants } from '../utils';
import { MotionCard } from '../components/common';
import { ScrollIndicator } from '../components/ui';

// Animation variants
const fadeInUp = motionVariants.fadeInUp();
const staggerContainer = motionVariants.stagger();

// Post card component
const PostCard = ({ post }) => {
  const categoryConfig = BLOG_CONFIG.categories[post.category];
  const withPublicUrl = (p) => {
    if (!p) return '';
    const base = process.env.PUBLIC_URL || '';
    if (p.startsWith('http')) return p;
    if (p.startsWith('/')) return `${base}${p}`;
    return `${base}/${p}`;
  };
  const headerImage = withPublicUrl(post.headerImage || `/blog/headers/default-${post.category}.jpg`);
  
  return (
    <MotionCard 
      className="bg-white dark:bg-gray-800 rounded-xl shadow-lg overflow-hidden border border-gray-100 dark:border-gray-700 h-full flex flex-col group hover:shadow-xl transition-all duration-200 mobile-card"
      hover="lift"
      variants={fadeInUp}
    >
      {/* Header Image */}
      <div className="relative overflow-hidden aspect-[16/9]">
        <img 
          src={headerImage}
          alt={post.title}
          className="w-full h-full object-cover transform transition-transform duration-700 group-hover:scale-105"
          onError={(e) => {
            const fallbackByCat = withPublicUrl(`/blog/headers/default-${post.category}.jpg`);
            const fallback = withPublicUrl('/blog/headers/default.jpg');
            if (e.target.src !== fallbackByCat) {
              e.target.src = fallbackByCat;
            } else if (e.target.src !== fallback) {
              e.target.src = fallback;
            }
          }}
        />
        <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-black/20 to-transparent"></div>
        
        {/* Category badge */}
        <div className="absolute top-4 left-4">
          <span 
            className={`inline-flex items-center px-3 py-1 rounded-full text-xs font-medium
              ${categoryConfig?.color === 'blue' ? 'bg-blue-100 text-blue-800 dark:bg-blue-900/50 dark:text-blue-300' : ''}
              ${categoryConfig?.color === 'indigo' ? 'bg-indigo-100 text-indigo-800 dark:bg-indigo-900/50 dark:text-indigo-300' : ''}
            `}
          >
            {categoryConfig?.color === 'blue' ? <Brain size={12} className="mr-1" /> : <FileText size={12} className="mr-1" />}
            {categoryConfig?.name || post.category}
          </span>
        </div>
      </div>
      
      {/* Content */}
      <div className="p-6 flex-1 flex flex-col">
        <div className="flex-1">
          <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-3 line-clamp-2 group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors">
            <Link to={`/blog/${post.category}/${post.slug}`}>
              {post.title}
            </Link>
          </h2>
          
          <p className="text-gray-600 dark:text-gray-300 mb-4 line-clamp-3 leading-relaxed">
              <p className="text-gray-600 dark:text-gray-300 mb-4 line-clamp-3 leading-relaxed card-description">{post.excerpt}</p>
          </p>
          
          {/* Tags */}
          {post.tags && post.tags.length > 0 && (
            <div className="flex flex-wrap gap-2 mb-4">
              {post.tags.slice(0, 3).map((tag, index) => (
                  <Link
                    key={index}
                    to={`/blog/tag/${encodeURIComponent(tag)}`}
                    className="card-tag inline-flex items-center"
                  >
                  <Tag size={10} className="mr-1" />
                  {tag}
                </Link>
              ))}
              {post.tags.length > 3 && (
                <span className="card-tag inline-flex items-center">
                  +{post.tags.length - 3} more
                </span>
              )}
            </div>
          )}
        </div>
        
        {/* Meta info */}
        <div className="flex items-center justify-between text-sm text-gray-500 dark:text-gray-400 mt-auto pt-4 border-t border-gray-100 dark:border-gray-700">
          <div className="flex items-center space-x-4">
            <div className="flex items-center">
              <Calendar size={14} className="mr-1" />
              <span>{formatDate(post.date, 'MMM d, yyyy')}</span>
            </div>
            <div className="flex items-center">
              <Clock size={14} className="mr-1" />
              <span>{post.readingTime} min read</span>
            </div>
          </div>
          
          <Link 
            to={`/blog/${post.category}/${post.slug}`}
            className="flex items-center text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300 transition-colors group"
          >
            <span className="mr-1">Read more</span>
            <ArrowRight size={14} className="transform transition-transform group-hover:translate-x-1" />
          </Link>
        </div>
      </div>
  </MotionCard>
  );
};

export default function BlogHomePage() {
  const [posts, setPosts] = useState([]);
  const [filteredPosts, setFilteredPosts] = useState([]);
  const [tags, setTags] = useState([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [selectedTag, setSelectedTag] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  const { scrollY } = useScroll();
  const heroRef = useRef(null);
  
  // Pagination configuration
  const POSTS_PER_PAGE = 6;
  
  // Transform values for smoother parallax effects
  const heroOpacity = useTransform(scrollY, [100, 400], [1, 0.95]);
  const heroY = useTransform(scrollY, [0, 400], [0, -50]);
  
  // Load posts and tags on component mount
  useEffect(() => {
    async function loadBlogData() {
      try {
        setLoading(true);
        const [postsData, tagsData] = await Promise.all([
          loadAllPosts(),
          getAllTags()
        ]);
        
        setPosts(postsData);
        setFilteredPosts(postsData);
        setTags(tagsData);
      } catch (err) {
        setError('Failed to load blog posts. Please try again later.');
        console.error('Error loading blog data:', err);
      } finally {
        setLoading(false);
      }
    }
    
    loadBlogData();
  }, []);
  
  // Filter posts based on search and filters
  useEffect(() => {
    let filtered = [...posts];
    
    // Filter by category
    if (selectedCategory !== 'all') {
      filtered = filtered.filter(post => post.category === selectedCategory);
    }
    
    // Filter by tag
    if (selectedTag) {
      filtered = filtered.filter(post => 
        post.tags && post.tags.some(tag => 
          tag.toLowerCase() === selectedTag.toLowerCase()
        )
      );
    }
    
    // Filter by search term
    if (searchTerm) {
      const term = searchTerm.toLowerCase();
      filtered = filtered.filter(post => 
        post.title.toLowerCase().includes(term) ||
        post.excerpt.toLowerCase().includes(term) ||
        (post.tags && post.tags.some(tag => tag.toLowerCase().includes(term)))
      );
    }
    
    setFilteredPosts(filtered);
    // Reset to first page when filters change
    setCurrentPage(1);
  }, [posts, searchTerm, selectedCategory, selectedTag]);
  
  // Pagination calculations
  const totalPosts = filteredPosts.length;
  const totalPages = Math.ceil(totalPosts / POSTS_PER_PAGE);
  const paginatedPosts = filteredPosts.slice((currentPage - 1) * POSTS_PER_PAGE, currentPage * POSTS_PER_PAGE);
  
  if (loading) {
    return (
      <div className="min-h-screen bg-white dark:bg-gray-900 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600 dark:text-gray-400">Loading blog posts...</p>
        </div>
      </div>
    );
  }
  
  if (error) {
    return (
      <div className="min-h-screen bg-white dark:bg-gray-900 flex items-center justify-center">
        <div className="text-center max-w-md">
          <BookOpen className="mx-auto h-12 w-12 text-gray-400 mb-4" />
          <h1 className="text-xl font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Unable to Load Blog
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mb-4">{error}</p>
          <button 
            onClick={() => window.location.reload()}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            Try Again
          </button>
        </div>
      </div>
    );
  }
  
  return (
    <>
      <Helmet>
        <title>Blog | Juan Lara</title>
        <meta name="description" content="Articles and insights about AI, machine learning, and technology by Juan Lara." />
        <meta property="og:title" content="Blog | Juan Lara" />
        <meta property="og:description" content="AI & Technology blog" />
      </Helmet>
      <div className="bg-white dark:bg-gray-900 text-gray-900 dark:text-gray-100 min-h-screen">
      
      {/* Hero Section + Scroll Indicator Container */}
      <div className="h-[calc(100dvh-5.5rem)] flex flex-col">
        {/* Hero Section */}
        <motion.section 
          ref={heroRef}
          style={{ opacity: heroOpacity, y: heroY }}
          className="relative flex-1 flex items-center justify-center overflow-hidden parallax-smooth"
        >
        <div className="absolute inset-0 bg-gradient-to-b from-blue-50 to-white dark:from-gray-800 dark:to-gray-900 -z-10"></div>
        
        {/* Decorative elements */}
        <div className="absolute top-40 right-20 w-72 h-72 rounded-full bg-blue-100/50 dark:bg-blue-900/20 blur-3xl -z-10"></div>
        <div className="absolute -bottom-20 -left-20 w-80 h-80 rounded-full bg-indigo-100/30 dark:bg-indigo-900/10 blur-3xl -z-10"></div>
        
  <div className="container mx-auto px-6 -mt-10 mobile-card-container">
          <motion.div 
            initial="hidden"
            animate="visible"
            variants={staggerContainer}
            className="max-w-4xl mx-auto text-center mb-0"
          >
            <motion.div variants={fadeInUp} className="mb-2">
              <div className="inline-flex items-center px-3 py-1 rounded-full bg-blue-100 text-blue-800 dark:bg-blue-900/50 dark:text-blue-300 text-sm font-medium mb-2">
                <BookOpen size={14} className="mr-1.5" /> Blog
              </div>
            </motion.div>
            
            <motion.h1 
              variants={fadeInUp}
              className="text-4xl md:text-5xl font-bold mb-2 leading-tight"
            >
              <span className="bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-indigo-600 dark:from-blue-400 dark:to-indigo-400">
                Thoughts & Discoveries
              </span>
            </motion.h1>
            
            <motion.p 
              variants={fadeInUp}
              className="text-lg md:text-xl text-gray-600 dark:text-gray-300 mb-4 max-w-3xl mx-auto"
            >
              Exploring mathematical curiosities, research insights, and the fascinating intersections between theory and practice.
            </motion.p>
            
            {/* Search and Filters */}
            <motion.div 
              variants={fadeInUp}
              className="flex flex-col md:flex-row gap-4 max-w-4xl mx-auto mb-8"
            >
              {/* Search */}
              <div className="relative flex-grow">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <Search size={18} className="text-gray-400" />
                </div>
                <input
                  type="text"
                  className="block w-full pl-10 pr-3 py-2.5 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-blue-500 dark:focus:ring-blue-400 focus:border-transparent"
                  placeholder="Search posts..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                />
              </div>
              
              {/* Category Filter */}
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <Filter size={18} className="text-gray-400" />
                </div>
                <select
                  className="block w-full pl-10 pr-10 py-2.5 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-blue-500 dark:focus:ring-blue-400 focus:border-transparent appearance-none"
                  value={selectedCategory}
                  onChange={(e) => setSelectedCategory(e.target.value)}
                >
                  <option value="all">All Categories</option>
                  {Object.entries(BLOG_CONFIG.categories).map(([key, config]) => (
                    <option key={key} value={key}>{config.name}</option>
                  ))}
                </select>
              </div>
              
              {/* Tag Filter */}
              {tags.length > 0 && (
                <div className="relative">
                  <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                    <Tag size={18} className="text-gray-400" />
                  </div>
                  <select
                    className="block w-full pl-10 pr-10 py-2.5 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-blue-500 dark:focus:ring-blue-400 focus:border-transparent appearance-none"
                    value={selectedTag}
                    onChange={(e) => setSelectedTag(e.target.value)}
                  >
                    <option value="">All Tags</option>
                    {tags.map(tag => (
                      <option key={tag} value={tag}>{tag}</option>
                    ))}
                  </select>
                </div>
              )}
            </motion.div>
          </motion.div>
        </div>
        
        </motion.section>
        
        {/* Scroll indicator */}
        <ScrollIndicator 
          fadeOutStart={0} 
          fadeOutEnd={200}
          size={22}
          className="flex-shrink-0"
        />
      </div>
      
      {/* Posts Grid */}
      <section className="py-0 pb-16">
        <div className="container mx-auto px-6 mobile-card-container">
          <motion.div 
            initial={false}
            whileInView="visible"
            viewport={{ once: true, margin: "0px" }}
            variants={staggerContainer}
            className="max-w-6xl mx-auto"
          >
            {/* Results info */}
            <motion.div 
              variants={fadeInUp}
              className="flex items-center justify-between mb-4"
            >
              <div className="flex items-center">
                <div className="w-12 h-12 rounded-full bg-blue-100 dark:bg-blue-900/30 flex items-center justify-center mr-4">
                  <Layers className="text-blue-600 dark:text-blue-400" size={24} />
                </div>
                <h2 className="text-3xl font-bold text-gray-900 dark:text-gray-100">
                  {selectedCategory !== 'all' 
                    ? BLOG_CONFIG.categories[selectedCategory]?.name 
                    : 'All Posts'}
                </h2>
              </div>
              
              <div className="text-gray-600 dark:text-gray-300">
                {totalPosts > 0 && (
                  <><span className="font-medium text-gray-700 dark:text-gray-100">{Math.min((currentPage - 1) * POSTS_PER_PAGE + 1, totalPosts)}</span>
                  {' '}-{' '}
                  <span className="font-medium text-gray-700 dark:text-gray-100">{Math.min(currentPage * POSTS_PER_PAGE, totalPosts)}</span>
                  {' '}of{' '}
                  <span className="font-medium text-gray-700 dark:text-gray-100">{totalPosts}</span> posts</>
                )}
              </div>
            </motion.div>
            
            {/* Category pills for easier filtering on desktop */}
            <motion.div 
              variants={fadeInUp}
              className="hidden lg:flex flex-wrap gap-3 mb-6"
            >
              <button
                key="all"
                onClick={() => setSelectedCategory('all')}
                aria-pressed={selectedCategory === 'all'}
                aria-label="Show all posts"
                className={`flex items-center px-4 py-2 rounded-full text-sm font-medium transition-colors
                  ${selectedCategory === 'all'
                    ? 'bg-blue-100 dark:bg-blue-900/50 text-blue-800 dark:text-blue-200 border border-blue-200 dark:border-blue-800' 
                    : 'bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700 border border-transparent'}`}
              >
                <Layers size={16} className="mr-2" />
                All Categories
              </button>
              {Object.entries(BLOG_CONFIG.categories).map(([key, config]) => (
                <button
                  key={key}
                  onClick={() => setSelectedCategory(key)}
                  aria-pressed={selectedCategory === key}
                  aria-label={`Filter posts by ${config.name}`}
                  className={`flex items-center px-4 py-2 rounded-full text-sm font-medium transition-colors
                    ${selectedCategory === key
                      ? `bg-${config.color}-100 dark:bg-${config.color}-900/50 text-${config.color}-800 dark:text-${config.color}-200 border border-${config.color}-200 dark:border-${config.color}-800` 
                      : 'bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700 border border-transparent'}`}
                >
                  {config.icon && (
                    <>
                      {typeof config.icon === 'function' ? (
                        <config.icon size={16} className="mr-2" />
                      ) : (
                        <BookOpen size={16} className="mr-2" />
                      )}
                    </>
                  )}
                  {config.name}
                </button>
              ))}
            </motion.div>
            
            {totalPosts === 0 ? (
              <motion.div 
                variants={fadeInUp}
                className="text-center py-16"
              >
                <div className="w-20 h-20 mx-auto mb-6 rounded-full bg-gray-100 dark:bg-gray-800 flex items-center justify-center">
                  <Search size={32} className="text-gray-400" />
                </div>
                <h3 className="text-xl font-semibold mb-2 text-gray-800 dark:text-gray-200">No posts found</h3>
                <p className="text-gray-600 dark:text-gray-400 mb-6">
                  Try adjusting your search or filters to find what you're looking for.
                </p>
                <button 
                  onClick={() => {
                    setSelectedCategory('all');
                    setSelectedTag('');
                    setSearchTerm('');
                  }}
                  className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors inline-flex items-center gap-2"
                >
                  <Layers size={16} />
                  Show all posts
                </button>
              </motion.div>
            ) : (
              <>
                <motion.div 
                  initial={false}
                  variants={staggerContainer}
                  className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 mobile-grid-single"
                >
                  {paginatedPosts.map(post => (
                    <PostCard key={`${post.category}-${post.slug}`} post={post} />
                  ))}
                </motion.div>
                
                {/* Pagination controls */}
                {totalPages > 1 && (
                  <motion.div
                    variants={fadeInUp}
                    className="flex flex-col sm:flex-row items-center justify-between mt-12 pt-8 border-t border-gray-200 dark:border-gray-700"
                  >
                    <div className="text-sm text-gray-600 dark:text-gray-400 mb-4 sm:mb-0">
                      <span className="font-medium text-gray-700 dark:text-gray-100">{Math.min((currentPage - 1) * POSTS_PER_PAGE + 1, totalPosts)}</span>
                      {' '}-{' '}
                      <span className="font-medium text-gray-700 dark:text-gray-100">{Math.min(currentPage * POSTS_PER_PAGE, totalPosts)}</span>
                      {' '}of{' '}
                      <span className="font-medium text-gray-700 dark:text-gray-100">{totalPosts}</span> posts
                    </div>
                    
                    <div className="flex items-center space-x-2">
                      <button
                        onClick={() => setCurrentPage(1)}
                        disabled={currentPage === 1}
                        className="px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors"
                      >
                        First
                      </button>
                      
                      <button
                        onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                        disabled={currentPage === 1}
                        className="px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors"
                      >
                        Previous
                      </button>
                      
                      <div className="px-3 text-sm text-gray-700 dark:text-gray-300">{currentPage} / {totalPages}</div>
                      
                      <button
                        onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
                        disabled={currentPage === totalPages}
                        className="px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors"
                      >
                        Next
                      </button>
                      
                      <button
                        onClick={() => setCurrentPage(totalPages)}
                        disabled={currentPage === totalPages}
                        className="px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors"
                      >
                        Last
                      </button>
                    </div>
                  </motion.div>
                )}
              </>
            )}
          </motion.div>
        </div>
      </section>
      
      {/* CTA Section */}
      <section className="py-16 bg-gray-900 dark:bg-gray-950">
        <div className="container mx-auto px-6 mobile-card-container">
          <motion.div 
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "0px" }}
            variants={staggerContainer}
            className="max-w-4xl mx-auto text-center text-white"
          >
            <motion.h2 
              variants={fadeInUp}
              className="text-3xl md:text-4xl font-bold mb-6"
            >
              Join the Conversation
            </motion.h2>
            
            <motion.p 
              variants={fadeInUp}
              className="text-gray-300 mb-8 max-w-2xl mx-auto"
            >
              Interested in discussing mathematical insights, AI developments, or research ideas? I'd love to hear your thoughts and explore new perspectives.
            </motion.p>
            
            <motion.div 
              variants={fadeInUp}
              className="flex flex-col sm:flex-row items-center justify-center gap-4"
            >
              <a 
                href="mailto:larajuand@outlook.com"
                className="px-8 py-3 bg-blue-500 hover:bg-blue-600 text-white rounded-lg transition-colors flex items-center gap-2 font-medium shadow-lg"
              >
                <Mail size={18} />
                <span>Share Your Ideas</span>
              </a>
              
              <a 
                href="https://www.linkedin.com/in/julara/"
                target="_blank"
                rel="noreferrer"
                className="px-8 py-3 bg-white text-gray-900 border border-gray-200 rounded-lg hover:bg-gray-100 transition-colors flex items-center gap-2 font-medium shadow-lg"
              >
                <svg 
                  className="w-5 h-5" 
                  viewBox="0 0 24 24" 
                  fill="currentColor"
                >
                  <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/>
                </svg>
                <span>Connect on LinkedIn</span>
              </a>
            </motion.div>
          </motion.div>
        </div>
      </section>
    </div>
    </>
  );
}