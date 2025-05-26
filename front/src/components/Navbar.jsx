import { useState, useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Sun, Moon, Menu } from 'lucide-react';
import MobileMenu from './MobileMenu';

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
    // Ensure body scrolling is restored when routes change
    document.body.style.overflow = 'auto';
  }, [location.pathname]);
  
  // Toggle mobile menu state
  const toggleMobileMenu = () => {
    const newMenuState = !mobileMenuOpen;
    setMobileMenuOpen(newMenuState);
    
    // Add/remove class to body for styling entire page
    if (newMenuState) {
      document.body.classList.add('menu-open');
      document.body.style.overflow = 'hidden';
    } else {
      document.body.classList.remove('menu-open');
      document.body.style.overflow = 'auto';
    }
  };
  
  // Get appropriate class for active/inactive links
  const getLinkClass = (path) => {
    const baseClass = "relative py-2 px-1 font-medium transition-all duration-300";
    const activeClass = "text-blue-600 dark:text-blue-400 before:absolute before:bottom-0 before:left-0 before:h-0.5 before:w-full before:bg-blue-600 dark:before:bg-blue-400";
    const inactiveClass = "text-slate-700 dark:text-slate-300 hover:text-blue-600 dark:hover:text-blue-400 before:absolute before:bottom-0 before:left-0 before:h-0.5 before:w-full before:scale-x-0 before:bg-blue-600 dark:before:bg-blue-400 before:origin-left before:transition-transform hover:before:scale-x-100";
    
    // Determine if this route is active
    const isActive = path === '/' 
      ? location.pathname === '/' 
      : location.pathname.startsWith(path);
    
    return `${baseClass} ${isActive ? activeClass : inactiveClass}`;
  };
  
  return (
    <>
      <header 
        className={`fixed w-full z-50 transition-all duration-500 ${
          scrolled 
            ? 'bg-white/95 dark:bg-gray-900/95 backdrop-blur-md shadow-lg py-3' 
            : 'bg-white/90 dark:bg-gray-900/90 backdrop-blur-sm py-4'
        }`}
      >
        <div className="container mx-auto px-4 sm:px-6">
          <div className="flex justify-between items-center">
            {/* Logo */}
            <motion.div 
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5 }}
              className="text-xl font-bold relative z-10"
            >
              <Link to="/" className="bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-indigo-600 dark:from-blue-400 dark:to-indigo-400">
                Juan Lara
              </Link>
              <div className="absolute -bottom-1 left-0 h-0.5 w-full bg-gradient-to-r from-blue-600 to-indigo-600 dark:from-blue-400 dark:to-indigo-400"></div>
            </motion.div>
            
            {/* Desktop Navigation */}
            <motion.nav 
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.1 }}
              className="hidden md:flex items-center space-x-8"
            >
              <Link to="/" className={getLinkClass('/')}>
                Home
              </Link>
              <Link to="/about" className={getLinkClass('/about')}>
                About
              </Link>
              <Link to="/projects" className={getLinkClass('/projects')}>
                Projects
              </Link>
              <Link to="/blog" className={getLinkClass('/blog')}>
                Blog
              </Link>
              
              {/* Theme Toggle Button */}
              <button 
                onClick={toggleDarkMode}
                aria-label="Toggle dark mode"
                className="p-2 rounded-full bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
              >
                {darkMode ? <Sun size={18} /> : <Moon size={18} />}
              </button>
            </motion.nav>
            
            {/* Mobile Menu Button */}
            <div className="md:hidden flex items-center gap-4 z-10">
              <button 
                onClick={toggleDarkMode}
                aria-label="Toggle dark mode"
                className="p-2 rounded-full bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
              >
                {darkMode ? <Sun size={18} /> : <Moon size={18} />}
              </button>
              
              <button
                onClick={toggleMobileMenu}
                aria-label="Toggle menu"
                className="p-2 rounded-lg text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
              >
                <Menu size={22} />
              </button>
            </div>
          </div>
        </div>
      </header>
      
      {/* Mobile Menu Component */}
      <MobileMenu 
        isOpen={mobileMenuOpen} 
        onClose={toggleMobileMenu} 
      />
    </>
  );
};

export default Navbar;