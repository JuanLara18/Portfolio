import { useState, useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Sun, Moon, Menu, X } from 'lucide-react';

const Navbar = ({ darkMode, toggleDarkMode }) => {
  const [scrolled, setScrolled] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const location = useLocation();
  
  // Track scrolling to change navbar appearance
  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 50);
    };
    
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);
  
  // Close mobile menu when changing routes
  useEffect(() => {
    setMobileMenuOpen(false);
  }, [location.pathname]);
  
  // Handle mobile menu toggle
  const toggleMobileMenu = () => {
    setMobileMenuOpen(!mobileMenuOpen);
    
    // Prevent background scrolling when menu is open
    if (!mobileMenuOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = 'auto';
    }
  };
  
  // Determine if a nav link is active
  const isActive = (path) => {
    if (path === '/') {
      return location.pathname === '/';
    }
    return location.pathname.startsWith(path);
  };
  
  // Get appropriate class for active/inactive links
  const getLinkClass = (path) => {
    const baseClass = "relative py-2 px-1 font-medium transition duration-300";
    const activeClass = "text-primary-600 dark:text-primary-500 before:absolute before:bottom-0 before:left-0 before:h-0.5 before:w-full before:bg-primary-600 dark:before:bg-primary-500";
    const inactiveClass = "text-slate-700 dark:text-slate-300 hover:text-primary-600 dark:hover:text-primary-500 before:absolute before:bottom-0 before:left-0 before:h-0.5 before:w-full before:scale-x-0 before:bg-primary-600 dark:before:bg-primary-500 before:origin-left before:transition-transform hover:before:scale-x-100";
    
    return `${baseClass} ${isActive(path) ? activeClass : inactiveClass}`;
  };
  
  return (
    <header className={`fixed w-full z-50 transition-all duration-500 ${scrolled ? 'bg-white/90 dark:bg-gray-900/95 backdrop-blur-md shadow-lg py-3' : 'py-5 backdrop-blur-0'}`}>
      <div className="container mx-auto px-6">
        <div className="flex justify-between items-center">
          {/* Logo */}
          <motion.div 
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5 }}
            className="text-xl font-bold relative"
          >
            <Link to="/" className="bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-indigo-600 dark:from-blue-400 dark:to-indigo-400">
              Juan Lara
            </Link>
            <div className="absolute -bottom-1 left-0 h-0.5 w-full bg-gradient-to-r from-blue-600 to-indigo-600 dark:from-blue-400 dark:to-indigo-400"></div>
          </motion.div>
          
          {/* Desktop Navigation */}
          <motion.nav 
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5 }}
            className="hidden md:flex items-center space-x-8"
          >
            <Link to="/" className={getLinkClass('/')}>Home</Link>
            <Link to="/about" className={getLinkClass('/about')}>About</Link>
            <Link to="/projects" className={getLinkClass('/projects')}>Projects</Link>
            <a 
              href="https://blog.juanlara.dev" 
              target="_blank" 
              rel="noopener noreferrer"
              className="relative py-2 px-1 font-medium transition duration-300 text-gray-700 dark:text-gray-300 hover:text-blue-600 dark:hover:text-blue-400 flex items-center gap-1"
            >
              Blog
              <svg 
                xmlns="http://www.w3.org/2000/svg" 
                width="14" 
                height="14" 
                viewBox="0 0 24 24" 
                fill="none" 
                stroke="currentColor" 
                strokeWidth="2" 
                strokeLinecap="round" 
                strokeLinejoin="round"
              >
                <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path>
                <polyline points="15 3 21 3 21 9"></polyline>
                <line x1="10" y1="14" x2="21" y2="3"></line>
              </svg>
            </a>
            
            {/* Theme Toggle Button */}
            <button 
              onClick={toggleDarkMode}
              aria-label="Toggle dark mode"
              className="p-2 rounded-full hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
            >
              {darkMode ? <Sun size={18} /> : <Moon size={18} />}
            </button>
          </motion.nav>
          
          {/* Mobile Menu Button */}
          <div className="md:hidden flex items-center gap-4">
            <button 
              onClick={toggleDarkMode}
              aria-label="Toggle dark mode"
              className="p-2 rounded-full hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
            >
              {darkMode ? <Sun size={18} /> : <Moon size={18} />}
            </button>
            
            <button
              onClick={toggleMobileMenu}
              aria-label="Toggle menu"
              className="text-gray-700 dark:text-gray-300"
            >
              {mobileMenuOpen ? <X size={24} /> : <Menu size={24} />}
            </button>
          </div>
        </div>
      </div>
      
      {/* Mobile Menu (slide out) */}
      <div 
        className={`fixed inset-0 bg-gray-900/50 backdrop-blur-sm z-40 md:hidden transition-opacity duration-300 
          ${mobileMenuOpen ? 'opacity-100' : 'opacity-0 pointer-events-none'}`}
        onClick={toggleMobileMenu}
      ></div>
      
      <div 
        className={`fixed top-0 right-0 w-64 h-full bg-white dark:bg-gray-900 z-50 transform transition-transform duration-300 ease-in-out shadow-2xl md:hidden
          ${mobileMenuOpen ? 'translate-x-0' : 'translate-x-full'}`}
      >
        <div className="p-6">
          <div className="flex justify-end mb-8">
            <button
              onClick={toggleMobileMenu}
              aria-label="Close menu"
              className="text-gray-700 dark:text-gray-300"
            >
              <X size={24} />
            </button>
          </div>
          
          <nav className="flex flex-col space-y-6">
            <Link to="/" className={`text-lg ${isActive('/') ? 'text-blue-600 dark:text-blue-400 font-medium' : 'text-gray-800 dark:text-gray-200'}`}>
              Home
            </Link>
            <Link to="/about" className={`text-lg ${isActive('/about') ? 'text-blue-600 dark:text-blue-400 font-medium' : 'text-gray-800 dark:text-gray-200'}`}>
              About
            </Link>
            <Link to="/projects" className={`text-lg ${isActive('/projects') ? 'text-blue-600 dark:text-blue-400 font-medium' : 'text-gray-800 dark:text-gray-200'}`}>
              Projects
            </Link>
            <a 
              href="https://blog.juanlara.dev" 
              target="_blank" 
              rel="noopener noreferrer"
              className="text-lg text-gray-800 dark:text-gray-200 flex items-center gap-2"
            >
              Blog
              <svg 
                xmlns="http://www.w3.org/2000/svg" 
                width="16" 
                height="16" 
                viewBox="0 0 24 24" 
                fill="none" 
                stroke="currentColor" 
                strokeWidth="2" 
                strokeLinecap="round" 
                strokeLinejoin="round"
              >
                <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path>
                <polyline points="15 3 21 3 21 9"></polyline>
                <line x1="10" y1="14" x2="21" y2="3"></line>
              </svg>
            </a>
          </nav>
          
          <div className="mt-8 pt-8 border-t border-gray-200 dark:border-gray-700">
            <div className="flex flex-col gap-4">
              <a 
                href="https://www.linkedin.com/in/julara/?locale=en_US"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 text-gray-800 dark:text-gray-200"
              >
                <svg 
                  xmlns="http://www.w3.org/2000/svg"
                  width="16" 
                  height="16" 
                  viewBox="0 0 24 24" 
                  fill="none" 
                  stroke="currentColor" 
                  strokeWidth="2" 
                  strokeLinecap="round" 
                  strokeLinejoin="round"
                >
                  <path d="M16 8a6 6 0 0 1 6 6v7h-4v-7a2 2 0 0 0-2-2 2 2 0 0 0-2 2v7h-4v-7a6 6 0 0 1 6-6z"></path>
                  <rect x="2" y="9" width="4" height="12"></rect>
                  <circle cx="4" cy="4" r="2"></circle>
                </svg>
                LinkedIn
              </a>
              
              <a 
                href="https://github.com/JuanLara18"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 text-gray-800 dark:text-gray-200"
              >
                <svg 
                  xmlns="http://www.w3.org/2000/svg"
                  width="16" 
                  height="16" 
                  viewBox="0 0 24 24" 
                  fill="none" 
                  stroke="currentColor" 
                  strokeWidth="2" 
                  strokeLinecap="round" 
                  strokeLinejoin="round"
                >
                  <path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22"></path>
                </svg>
                GitHub
              </a>
              
              <a 
                href={`${process.env.PUBLIC_URL}/documents/CV___EN.pdf`}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 text-gray-800 dark:text-gray-200"
              >
                <svg 
                  xmlns="http://www.w3.org/2000/svg"
                  width="16" 
                  height="16" 
                  viewBox="0 0 24 24" 
                  fill="none" 
                  stroke="currentColor" 
                  strokeWidth="2" 
                  strokeLinecap="round" 
                  strokeLinejoin="round"
                >
                  <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                  <polyline points="14 2 14 8 20 8"></polyline>
                  <line x1="16" y1="13" x2="8" y2="13"></line>
                  <line x1="16" y1="17" x2="8" y2="17"></line>
                  <polyline points="10 9 9 9 8 9"></polyline>
                </svg>
                Resume
              </a>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Navbar;