import { useState, useEffect, useRef } from 'react';
import { motion, useScroll, useTransform, useInView } from 'framer-motion';
import { 
  ExternalLink, 
  Github, 
  Code, 
  Database, 
  BarChart, 
  Layers, 
  Box, 
  Cpu, 
  Gamepad, 
  Server, 
  Globe, 
  FileText, 
  Music, 
  Search, 
  Filter, 
  Terminal,
  Mail
} from 'lucide-react';

// Animation variants
const fadeInUp = {
  hidden: { opacity: 0, y: 30 },
  visible: { 
    opacity: 1, 
    y: 0,
    transition: { duration: 0.7, ease: [0.22, 1, 0.36, 1] }
  }
};

const fadeInRight = {
  hidden: { opacity: 0, x: -30 },
  visible: { 
    opacity: 1, 
    x: 0,
    transition: { duration: 0.7, ease: [0.22, 1, 0.36, 1] }
  }
};

const staggerContainer = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.15,
      delayChildren: 0.2
    }
  }
};

const cardHover = {
  rest: { scale: 1, y: 0 },
  hover: { 
    scale: 1.02, 
    y: -8,
    transition: {
      duration: 0.4,
      ease: [0.22, 1, 0.36, 1]
    }
  }
};

// Project categories with their icons and colors
const categories = [
  { id: 'all', name: 'All Projects', icon: Layers, color: 'blue' },
  { id: 'ml', name: 'Machine Learning & AI', icon: Brain, color: 'purple' },
  { id: 'web', name: 'Web Development', icon: Globe, color: 'indigo' },
  { id: 'data', name: 'Data Science', icon: BarChart, color: 'green' },
  { id: 'tools', name: 'Tools & Utilities', icon: Terminal, color: 'yellow' },
  { id: 'games', name: 'Games & Interactive', icon: Gamepad, color: 'red' },
  { id: 'upcoming', name: 'Upcoming Projects', icon: FileText, color: 'teal' }
];

// Brain icon is not imported, so let's define it
function Brain(props) {
  return (
    <svg 
      xmlns="http://www.w3.org/2000/svg" 
      width={props.size || 24} 
      height={props.size || 24} 
      viewBox="0 0 24 24" 
      fill="none" 
      stroke="currentColor" 
      strokeWidth="2" 
      strokeLinecap="round" 
      strokeLinejoin="round" 
      className={props.className}
    >
      <path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 4.44-2.04Z"></path>
      <path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96.44 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24A2.5 2.5 0 0 0 14.5 2Z"></path>
    </svg>
  );
}

// Project data based on the GitHub repositories
const projects = [
  {
    id: 1,
    name: "TextInsight",
    description: "Advanced text analysis library combining BERT for sentiment analysis, GPT-3.5 for text correction and topic generation, and embeddings for graph visualization. Deployed at Ipsos to analyze survey data, cutting analysis time by 60% while delivering deeper insights.",
    image: "textinsight.png",
    tags: ["NLP", "Transformers", "OpenAI", "NetworkX", "PyVis"],
    github: "https://github.com/JuanLara18/TextInsight",
    demo: "https://textinsight-ipsos.streamlit.app/",
    category: "ml",
    featured: true
  },
  {
    id: 2,
    name: "Meeting-Scribe",
    description: "AI-powered meeting transcription tool with speaker diarization using Whisper and pyannote-audio. Automatically transcribes meetings, identifies speakers, and generates summaries.",
    image: "meeting-scribe.png",
    tags: ["ffmpeg", "transcription", "whisper", "pyannote-audio", "diarization"],
    github: "https://github.com/JuanLara18/Meeting-Scribe",
    category: "ml",
    featured: true
  },
  {
    id: 3,
    name: "Translation",
    description: "High-performance distributed translation system for large multilingual datasets using PySpark and OpenAI. Supports caching, checkpointing, and metadata-preserving Stata translation.",
    image: "translation.png",
    tags: ["nlp", "distributed-computing", "stata", "pyspark", "openai"],
    github: "https://github.com/JuanLara18/Translation",
    category: "ml",
    featured: true
  },
  {
    id: 4,
    name: "AgentFlow",
    description: "Simulation framework to visualize multi-agent organizational dynamics using a modular Streamlit-based interface. Models organizational behavior and hierarchies.",
    image: "agentflow.png",
    tags: ["simulation", "multi-agent-systems", "organizational-model"],
    github: "https://github.com/JuanLara18/AgentFlow",
    category: "ml"
  },
  {
    id: 5,
    name: "Classification",
    description: "Modular pipeline for text clustering, classification, and evaluation using TF-IDF and unsupervised ML techniques. Optimized for large-scale document processing.",
    image: "classification.png",
    tags: ["nlp", "unsupervised-learning", "tfidf", "text-clustering"],
    github: "https://github.com/JuanLara18/Classification",
    category: "ml"
  },
  {
    id: 6,
    name: "Pharmacy Segmentation Application",
    description: "Responsive mobile field application for pharmacy segmentation using R Shiny with Google Cloud Storage integration. Features geolocation mapping with Leaflet, route management, and real-time ML classification, reducing manual segmentation effort by 85%.",
    image: "pharmacy-app.png",
    tags: ["R Shiny", "Google Cloud", "Random Forest", "Leaflet"],
    category: "data",
    featured: true
  },
  {
    id: 7,
    name: "QuizApp",
    description: "A modern full-stack learning platform that lets educators create and manage quizzes, and delivers instant feedback to learners via a React frontend and Flask backend.",
    image: "quizapp.png",
    tags: ["flask", "reactjs", "sqlite", "quiz"],
    github: "https://github.com/JuanLara18/QuizApp",
    category: "web"
  },
  {
    id: 8,
    name: "Whiteboard-app",
    description: "Collaborative whiteboard built with React, TypeScript, and WebSockets. Draw, share, and brainstorm ideas in real-time with team members.",
    image: "whiteboard.png",
    tags: ["react", "real-time", "typescript", "whiteboard"],
    github: "https://github.com/JuanLara18/whiteboard-app",
    category: "web"
  },
  {
    id: 9,
    name: "Cunservicios Platform",
    description: "Smart platform for managing public utility services, including bill consultation, claims (PQR), and payment tracking. Built with React and Python.",
    image: "cunservicios.png",
    tags: ["react", "python", "sql", "pqr"],
    github: "https://github.com/JuanLara18/Cunservicios",
    category: "web",
    featured: true
  },
  {
    id: 10,
    name: "Notebook-Converter",
    description: "Easily convert Jupyter Notebooks to different formats (PDF, HTML, Markdown) using a customizable and efficient command-line tool. Optimized for data scientists and researchers.",
    image: "notebook-converter.png",
    tags: ["automation", "jupyter-notebook"],
    github: "https://github.com/JuanLara18/Notebook-Converter",
    category: "tools"
  },
  {
    id: 11,
    name: "AI-Roadmap",
    description: "A structured, project-based roadmap for learning Machine Learning & AI through hands-on projects. Each project builds on the previous one, helping you move from beginner to advanced AI concepts.",
    image: "ai-roadmap.png",
    tags: ["ai", "ml", "education"],
    github: "https://github.com/JuanLara18/AI-Roadmap",
    category: "ml"
  },
  {
    id: 12,
    name: "MadameX",
    description: "Interactive cryptography toolkit that allows users to encrypt and decrypt messages using various algorithms, as well as perform cryptanalysis to break encrypted messages without the original key.",
    image: "madamex.png",
    tags: ["cryptography", "cryptoanalysis", "encryption-decryption"],
    github: "https://github.com/JuanLara18/MadameX",
    demo: "https://juanlara18.github.io/MadameX/",
    category: "web"
  },
  {
    id: 13,
    name: "BrickBreaker",
    description: "Classic brick breaker game built with JavaScript and p5.js. Features multiple levels, power-ups, and increasing difficulty.",
    image: "brickbreaker.png",
    tags: ["object-oriented-programming", "retrogames", "JavaScript"],
    github: "https://github.com/JuanLara18/BrickBreaker",
    category: "games"
  },
  {
    id: 14,
    name: "Tetris",
    description: "Implementation of the classic Tetris game using JavaScript and p5 library, developed for the Object-oriented Programming course.",
    image: "tetris.png", 
    tags: ["object-oriented-programming", "retrogames", "JavaScript"],
    github: "https://github.com/JuanLara18/Tetris",
    category: "games"
  },
  // Upcoming/In-progress projects
  {
    id: 15,
    name: "BalanceAI",
    description: "Reinforcement learning project to teach an AI agent to balance itself. Visual interface allows watching the agent learn in real-time as it improves through training iterations.",
    image: "balance-ai.png",
    tags: ["reinforcement-learning", "machine-learning", "python", "gymnasium"],
    category: "upcoming"
  },
  {
    id: 16,
    name: "DJ-Mix Generator",
    description: "AI-powered DJ that creates seamless transitions between songs based on your preferences. Analyzes BPM, key, and energy levels to create professional sounding mixes with techno focus.",
    image: "dj-mix.png",
    tags: ["audio-processing", "machine-learning", "music-generation", "python"],
    category: "upcoming"
  },
  {
    id: 17,
    name: "FoodEconomy",
    description: "Web app that tracks historical prices of essential household products, analyzes their evolution over time, and recommends optimized shopping lists based on price trends and user preferences.",
    image: "food-economy.png",
    tags: ["web-scraping", "data-analysis", "price-prediction", "react", "python"],
    category: "upcoming",
    featured: true
  }
];

// Project card component
const ProjectCard = ({ project, inView }) => {
  const getIcon = (tag) => {
    switch (tag.toLowerCase()) {
      case 'ml': case 'ai': case 'machine-learning':
        return <Brain size={14} />;
      case 'python': case 'flask': case 'javascript': case 'typescript': case 'react':
        return <Code size={14} />;
      case 'data': case 'data-analysis': case 'price-prediction':
        return <BarChart size={14} />;
      case 'nlp': case 'transformers': case 'openai':
        return <Terminal size={14} />;
      default:
        return <Box size={14} />;
    }
  };

  // Get category color
  const getCategoryColor = (categoryId) => {
    const category = categories.find(c => c.id === categoryId);
    return category ? category.color : 'gray';
  };

  
  return (
    <motion.div 
      className={`bg-white dark:bg-gray-800 rounded-xl shadow-lg overflow-hidden border border-gray-100 dark:border-gray-700 h-full flex flex-col
      ${project.featured ? 'col-span-2' : ''}`}
      initial="rest"
      whileHover="hover"
      variants={cardHover}
    >
      <div className="relative overflow-hidden aspect-[4/3]">
        <div className="absolute inset-0 bg-gradient-to-br from-blue-500/20 to-purple-500/20 z-10"></div>
        {project.image ? (
          <img 
            src={`/images/project-previews/${project.image}`} 
            alt={project.name} 
            className="w-full h-full object-cover transform transition-transform duration-700 hover:scale-105"
          />
        ) : (
          <div className={`w-full h-full flex items-center justify-center bg-${getCategoryColor(project.category)}-100 dark:bg-${getCategoryColor(project.category)}-900/30`}>
            {(() => {
              const category = categories.find(c => c.id === project.category);
              if (category && typeof category.icon === 'function') {
                const IconComponent = category.icon;
                return <IconComponent size={48} className={`text-${getCategoryColor(project.category)}-600 dark:text-${getCategoryColor(project.category)}-400 opacity-40`} />;
              }
              return <Box size={48} className={`text-${getCategoryColor(project.category)}-600 dark:text-${getCategoryColor(project.category)}-400 opacity-40`} />;
            })()}
          </div>
        )}
        
        {/* Category badge */}
        <div className={`absolute top-4 left-4 z-20 bg-${getCategoryColor(project.category)}-100 dark:bg-${getCategoryColor(project.category)}-900/50 text-${getCategoryColor(project.category)}-800 dark:text-${getCategoryColor(project.category)}-200 text-xs font-medium px-2.5 py-1 rounded-full flex items-center gap-1`}>
          {/* Use conditional rendering to safely render the icon */}
          {(() => {
            const category = categories.find(c => c.id === project.category);
            if (category && typeof category.icon === 'function') {
              const IconComponent = category.icon;
              return <IconComponent size={12} />;
            }
            return <Box size={12} />;
          })()}
          <span>{categories.find(c => c.id === project.category)?.name}</span>
        </div>
        
        {project.featured && (
          <div className="absolute top-4 right-4 z-20 bg-yellow-100 dark:bg-yellow-900/50 text-yellow-800 dark:text-yellow-200 text-xs font-medium px-2.5 py-1 rounded-full">
            Featured
          </div>
        )}
      </div>
      
      <div className="p-6">
        <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-2">{project.name}</h3>
        <p className="text-gray-600 dark:text-gray-300 mb-4 line-clamp-3">{project.description}</p>
        
        <div className="flex flex-wrap gap-1.5 mb-6">
          {project.tags && project.tags.slice(0, 4).map((tag, index) => (
            <span 
              key={index} 
              className="inline-flex items-center px-2 py-1 text-xs font-medium rounded bg-gray-100 dark:bg-gray-700 text-gray-800 dark:text-gray-200 gap-1"
            >
              {getIcon(tag)}
              {tag}
            </span>
          ))}
          {project.tags && project.tags.length > 4 && (
            <span className="inline-flex items-center px-2 py-1 text-xs font-medium rounded bg-gray-100 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
              +{project.tags.length - 4} more
            </span>
          )}
        </div>
        
        <div className="flex items-center justify-between mt-auto">
          <div className="flex items-center gap-3">
            {project.github && (
              <a 
                href={project.github} 
                target="_blank" 
                rel="noopener noreferrer"
                className="text-gray-600 dark:text-gray-300 hover:text-blue-600 dark:hover:text-blue-400 transition-colors"
                aria-label={`GitHub repository for ${project.name}`}
              >
                <Github size={20} />
              </a>
            )}
            {project.demo && (
              <a 
                href={project.demo} 
                target="_blank" 
                rel="noopener noreferrer"
                className="text-gray-600 dark:text-gray-300 hover:text-blue-600 dark:hover:text-blue-400 transition-colors"
                aria-label={`Live demo for ${project.name}`}
              >
                <ExternalLink size={20} />
              </a>
            )}
          </div>
          
          {project.category === 'upcoming' ? (
            <span className="text-xs font-medium bg-indigo-100 dark:bg-indigo-900/40 text-indigo-700 dark:text-indigo-300 px-2.5 py-1 rounded-full">
              Coming Soon
            </span>
          ) : null}
        </div>
      </div>
    </motion.div>
  );
};

// Main Projects Page Component
export default function ProjectsPage() {
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [searchTerm, setSearchTerm] = useState('');
  const [filteredProjects, setFilteredProjects] = useState(projects);
  const { scrollY } = useScroll();
  const heroRef = useRef(null);
  const featuredRef = useRef(null);
  const isFeaturedInView = useInView(featuredRef, { once: true, margin: "-100px" });
  
  // Transform values based on scroll position
  const heroOpacity = useTransform(scrollY, [0, 300], [1, 0.6]);
  const heroScale = useTransform(scrollY, [0, 300], [1, 0.95]);
  
  // Filter projects based on category and search term with optimized visual layout
  useEffect(() => {
    let filtered = [...projects];
    
    // Apply category filter first
    if (selectedCategory !== 'all') {
      filtered = filtered.filter(project => project.category === selectedCategory);
    }
    
    // Apply search filter if there's a search term
    if (searchTerm) {
      const term = searchTerm.toLowerCase();
      filtered = filtered.filter(project => 
        project.name.toLowerCase().includes(term) || 
        project.description.toLowerCase().includes(term) ||
        (project.tags && project.tags.some(tag => tag.toLowerCase().includes(term)))
      );
    }
    
    // Only reorganize if showing all projects without search
    if (selectedCategory === 'all' && !searchTerm) {
      const featuredProjects = filtered.filter(p => p.featured);
      const regularProjects = filtered.filter(p => !p.featured);
      
      // Create grid-aware layout that fills space efficiently
      const createOptimalLayout = () => {
        // Constants for our grid layouts
        const DESKTOP_COLUMNS = 4;
        const MOBILE_COLUMNS = 2;
        // A featured project takes 2 columns in both layouts
        const FEATURED_WIDTH = 2;
        
        // Calculate how many slots we have to fill
        const totalRegularSlots = regularProjects.length;
        const totalFeaturedSlots = featuredProjects.length * FEATURED_WIDTH;
        const totalSlots = totalRegularSlots + totalFeaturedSlots;
        
        // Initialize arrays to track our grid
        const result = [];
        let currentGridState = []; // Represents the state of the current row being built
        let regularIndex = 0;
        let featuredIndex = 0;
        
        // Function to add a project to the result and update grid state
        const addToGrid = (project, width) => {
          result.push(project);
          
          // Update grid state (desktop layout)
          for (let i = 0; i < width; i++) {
            currentGridState.push(true);
          }
          
          // If we've filled a row or more, reset the grid state
          while (currentGridState.length >= DESKTOP_COLUMNS) {
            currentGridState = currentGridState.slice(DESKTOP_COLUMNS);
          }
        };
        
        // Distribute projects optimally across the grid
        while (regularIndex < regularProjects.length || featuredIndex < featuredProjects.length) {
          // Check if we can place a featured project in the current row
          const canPlaceFeatured = 
            featuredIndex < featuredProjects.length && 
            (currentGridState.length + FEATURED_WIDTH <= DESKTOP_COLUMNS);
          
          // Check if placing a featured project would minimize gaps
          const shouldPlaceFeatured = 
            canPlaceFeatured && 
            (
              // Place featured at start of row
              currentGridState.length === 0 ||
              // Place featured at end of row if it fits perfectly
              currentGridState.length === DESKTOP_COLUMNS - FEATURED_WIDTH ||
              // Place featured at strategic positions or if we're running out of regular projects
              regularIndex === regularProjects.length ||
              // Place featured after every ~6 regular projects for visual rhythm
              (featuredIndex < featuredProjects.length - 1 && 
              regularIndex > 0 && 
              regularIndex % 6 === 0)
            );
            
          if (shouldPlaceFeatured) {
            // Place a featured project
            addToGrid(featuredProjects[featuredIndex], FEATURED_WIDTH);
            featuredIndex++;
          } else if (regularIndex < regularProjects.length) {
            // Place a regular project
            addToGrid(regularProjects[regularIndex], 1);
            regularIndex++;
          } else if (featuredIndex < featuredProjects.length) {
            // Force place remaining featured projects at the beginning of the next row
            if (currentGridState.length !== 0) {
              // Fill the current row with regular projects if available
              while (currentGridState.length < DESKTOP_COLUMNS && regularIndex < regularProjects.length) {
                addToGrid(regularProjects[regularIndex], 1);
                regularIndex++;
              }
              // Reset to start a new row
              currentGridState = [];
            }
            
            // Now place the featured project at the start of a row
            addToGrid(featuredProjects[featuredIndex], FEATURED_WIDTH);
            featuredIndex++;
          }
        }
        
        // If we still have incomplete row, fill it with any remaining projects
        // This should rarely happen with the logic above, but just in case
        if (regularIndex < regularProjects.length && currentGridState.length > 0) {
          while (currentGridState.length < DESKTOP_COLUMNS && regularIndex < regularProjects.length) {
            addToGrid(regularProjects[regularIndex], 1);
            regularIndex++;
          }
        }
        
        return result;
      };
      
      filtered = createOptimalLayout();
    }
    
    setFilteredProjects(filtered);
  }, [selectedCategory, searchTerm, projects]);

  // Get featured projects
  const featuredProjects = projects.filter(project => project.featured);
  
  return (
    <div className="bg-white dark:bg-gray-900 text-gray-900 dark:text-gray-100 min-h-screen">
      {/* Hero Section */}
      <motion.section 
        ref={heroRef}
        style={{ opacity: heroOpacity, scale: heroScale }}
        className="relative pt-32 pb-20 md:pt-40 md:pb-32 overflow-hidden"
      >
        <div className="absolute inset-0 bg-gradient-to-b from-indigo-50 to-white dark:from-gray-800 dark:to-gray-900 -z-10"></div>
        
        {/* Decorative elements */}
        <div className="absolute top-40 right-20 w-72 h-72 rounded-full bg-blue-100/50 dark:bg-blue-900/20 blur-3xl -z-10"></div>
        <div className="absolute -bottom-20 -left-20 w-80 h-80 rounded-full bg-indigo-100/30 dark:bg-indigo-900/10 blur-3xl -z-10"></div>
        
        <div className="container mx-auto px-6">
          <motion.div 
            initial="hidden"
            animate="visible"
            variants={staggerContainer}
            className="max-w-4xl mx-auto text-center"
          >
            <motion.div variants={fadeInUp} className="mb-4">
              <div className="inline-flex items-center px-3 py-1 rounded-full bg-indigo-100 text-indigo-800 dark:bg-indigo-900/50 dark:text-indigo-300 text-sm font-medium mb-4">
                <Layers size={14} className="mr-1.5" /> Project Portfolio
              </div>
            </motion.div>
            
            <motion.h1 
              variants={fadeInUp}
              className="text-4xl md:text-5xl font-bold mb-6 leading-tight"
            >
              <span className="bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-indigo-600 dark:from-blue-400 dark:to-indigo-400">
                Exploring Solutions
              </span>
              <span className="block mt-1 text-gray-800 dark:text-gray-100">
                Through Code
              </span>
            </motion.h1>
            
            <motion.p 
              variants={fadeInUp}
              className="text-lg md:text-xl text-gray-600 dark:text-gray-300 mb-8 max-w-3xl mx-auto"
            >
              A showcase of my work in AI, machine learning, web development, and more. Each project represents a unique challenge solved through computational thinking and creative problem-solving.
            </motion.p>
            
            {/* Search and filter */}
            <motion.div 
              variants={fadeInUp}
              className="flex flex-col md:flex-row gap-4 max-w-xl mx-auto"
            >
              <div className="relative flex-grow">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <Search size={18} className="text-gray-400" />
                </div>
                <input
                  type="text"
                  className="block w-full pl-10 pr-3 py-2.5 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-blue-500 dark:focus:ring-blue-400 focus:border-transparent"
                  placeholder="Search projects..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                />
              </div>
              
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <Filter size={18} className="text-gray-400" />
                </div>
                <select
                  className="block w-full pl-10 pr-10 py-2.5 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-blue-500 dark:focus:ring-blue-400 focus:border-transparent appearance-none"
                  value={selectedCategory}
                  onChange={(e) => setSelectedCategory(e.target.value)}
                >
                  {categories.map(category => (
                    <option key={category.id} value={category.id}>
                      {category.name}
                    </option>
                  ))}
                </select>
                <div className="absolute inset-y-0 right-0 pr-3 flex items-center pointer-events-none">
                  <svg className="h-5 w-5 text-gray-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
                    <path fillRule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clipRule="evenodd" />
                  </svg>
                </div>
              </div>
            </motion.div>
          </motion.div>
        </div>
      </motion.section>
      
      {/* All Projects Grid */}
      <section className="py-16">
        <div className="container mx-auto px-6">
          <motion.div 
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-100px" }}
            variants={staggerContainer}
            className="max-w-6xl mx-auto"
          >
            <motion.div 
              variants={fadeInRight}
              className="flex items-center justify-between mb-10"
            >
              <div className="flex items-center">
                <div className="w-12 h-12 rounded-full bg-blue-100 dark:bg-blue-900/30 flex items-center justify-center mr-4">
                  <Layers className="text-blue-600 dark:text-blue-400" size={24} />
                </div>
                <h2 className="text-3xl font-bold text-gray-900 dark:text-gray-100">
                  {selectedCategory !== 'all' 
                    ? categories.find(c => c.id === selectedCategory)?.name 
                    : 'All Projects'}
                </h2>
              </div>
              
              <div className="text-gray-600 dark:text-gray-300">
                {filteredProjects.length} {filteredProjects.length === 1 ? 'project' : 'projects'} found
              </div>
            </motion.div>
            
            {/* Category pills for easier filtering on desktop */}
            <motion.div 
              variants={fadeInUp}
              className="hidden lg:flex flex-wrap gap-3 mb-10"
            >
              {categories.map(category => (
                <button
                  key={category.id}
                  onClick={() => setSelectedCategory(category.id)}
                  className={`flex items-center px-4 py-2 rounded-full text-sm font-medium transition-colors
                    ${selectedCategory === category.id 
                      ? `bg-${category.color}-100 dark:bg-${category.color}-900/50 text-${category.color}-800 dark:text-${category.color}-200 border border-${category.color}-200 dark:border-${category.color}-800` 
                      : 'bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700 border border-transparent'}`}
                >
                  {(() => {
                    if (typeof category.icon === 'function') {
                      const IconComponent = category.icon;
                      return <IconComponent size={16} className="mr-2" />;
                    }
                    return <Box size={16} className="mr-2" />;
                  })()}
                  {category.name}
                </button>
              ))}
            </motion.div>
            
            {filteredProjects.length === 0 ? (
              <motion.div 
                variants={fadeInUp}
                className="text-center py-16"
              >
                <div className="w-20 h-20 mx-auto mb-6 rounded-full bg-gray-100 dark:bg-gray-800 flex items-center justify-center">
                  <Search size={32} className="text-gray-400" />
                </div>
                <h3 className="text-xl font-semibold mb-2 text-gray-800 dark:text-gray-200">No projects found</h3>
                <p className="text-gray-600 dark:text-gray-400 mb-6">
                  Try adjusting your search or filter to find what you're looking for.
                </p>
                <button 
                  onClick={() => {
                    setSelectedCategory('all');
                    setSearchTerm('');
                  }}
                  className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors inline-flex items-center gap-2"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M3 12h18M3 6h18M3 18h18"></path>
                  </svg>
                  Show all projects
                </button>
              </motion.div>
            ) : (
              <motion.div 
                variants={fadeInUp}
                className="grid grid-cols-2 sm:grid-cols-2 lg:grid-cols-4 gap-5 md:gap-6 auto-rows-auto"
              >
                {filteredProjects.map(project => (
                  <ProjectCard key={project.id} project={project} inView={true} />
                ))}
              </motion.div>
            )}
          </motion.div>
        </div>
      </section>
      
      {/* Collaboration CTA */}
      <section className="py-24 bg-gradient-to-br from-blue-600 to-indigo-700 dark:from-blue-700 dark:to-indigo-900 text-white">
        <div className="container mx-auto px-6">
          <motion.div 
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-100px" }}
            variants={staggerContainer}
            className="max-w-4xl mx-auto text-center"
          >
            <motion.h2 
              variants={fadeInUp}
              className="text-3xl md:text-4xl font-bold mb-6"
            >
              Let's Build Something Amazing Together
            </motion.h2>
            
            <motion.p 
              variants={fadeInUp}
              className="text-lg md:text-xl text-blue-100 mb-10 max-w-3xl mx-auto"
            >
              Have a project idea or collaboration opportunity? I'm always interested in discussing new challenges and innovative solutions.
            </motion.p>
            
            <motion.div 
              variants={fadeInUp}
              className="flex flex-col sm:flex-row items-center justify-center gap-4"
            >
              <a 
                href="mailto:larajuand@outlook.com"
                className="px-8 py-3 bg-white text-blue-700 rounded-lg hover:bg-blue-50 transition-colors flex items-center gap-2 font-medium"
              >
                <Mail size={18} />
                <span>Get in Touch</span>
              </a>
              
              <a 
                href="https://github.com/JuanLara18"
                target="_blank"
                rel="noopener noreferrer"
                className="px-8 py-3 bg-blue-500 text-white border border-blue-400 rounded-lg hover:bg-blue-600 transition-colors flex items-center gap-2 font-medium"
              >
                <Github size={18} />
                <span>View GitHub</span>
              </a>
            </motion.div>
          </motion.div>
        </div>
      </section>
    </div>
  );
}