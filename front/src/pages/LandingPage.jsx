import { useState, useEffect, useRef } from 'react';
import { motion, useScroll, useTransform, useMotionValueEvent } from 'framer-motion';
import { ChevronDown, Github, Linkedin, Mail, ExternalLink, Code, Terminal, Database, Server, Cpu, TerminalSquare, FileCode, Braces, Layers } from 'lucide-react';
import { Link } from 'react-router-dom';

// Technical particle animation constants
const GRID_SIZE = 30;
const DOT_SIZE = 1;

// Animation variants
const fadeInUp = {
  visible: { 
    opacity: 1, 
    y: 0,
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

const slideInRight = {
  hidden: { opacity: 0, x: 40 },
  visible: { 
    opacity: 1, 
    x: 0,
    transition: { duration: 0.7, ease: [0.22, 1, 0.36, 1] }
  }
};

const slideInLeft = {
  hidden: { opacity: 0, x: -40 },
  visible: { 
    opacity: 1, 
    x: 0,
    transition: { duration: 0.7, ease: [0.22, 1, 0.36, 1] }
  }
};

const scaleUp = {
  hidden: { opacity: 0, scale: 0.9 },
  visible: { 
    opacity: 1, 
    scale: 1,
    transition: { duration: 0.6, ease: [0.22, 1, 0.36, 1] }
  }
};

// Terminal text typing animation component
const TypingTerminal = ({ text }) => {
  const [displayText, setDisplayText] = useState('');
  const [currentIndex, setCurrentIndex] = useState(0);
  const [showCursor, setShowCursor] = useState(true);

  useEffect(() => {
    if (currentIndex < text.length) {
      const timeout = setTimeout(() => {
        setDisplayText(prev => prev + text[currentIndex]);
        setCurrentIndex(prev => prev + 1);
      }, 60 + Math.random() * 40); // Varying typing speed for realism
      
      return () => clearTimeout(timeout);
    } else {
      // Blink cursor after typing is complete
      const blinkInterval = setInterval(() => {
        setShowCursor(prev => !prev);
      }, 530);
      
      return () => clearInterval(blinkInterval);
    }
  }, [currentIndex, text]);

  return (
    <div className="font-mono p-4 bg-gray-900 text-green-400 rounded-md overflow-hidden shadow-xl border border-gray-800">
      <div className="flex items-center gap-2 mb-2 pb-2 border-b border-gray-700">
        <div className="w-3 h-3 rounded-full bg-red-500" />
        <div className="w-3 h-3 rounded-full bg-yellow-500" />
        <div className="w-3 h-3 rounded-full bg-green-500" />
        <span className="ml-2 text-gray-400 text-sm">juan@lara ~ terminal</span>
      </div>
      <div className="flex">
        <span className="text-blue-400 mr-2">$</span>
        <div>
          {displayText}
          {showCursor && <span className="inline-block w-2 h-4 bg-green-400 ml-1 animate-pulse"></span>}
        </div>
      </div>
    </div>
  );
};

// Animated tech icon component
const TechIcon = ({ icon: Icon, label, delay = 0 }) => (
  <motion.div 
    className="flex flex-col items-center justify-center p-4"
    initial={{ opacity: 0, y: 20 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ delay: delay, duration: 0.5 }}
  >
    <div className="w-12 h-12 flex items-center justify-center rounded-lg bg-blue-50 dark:bg-blue-900/30 mb-2 text-blue-600 dark:text-blue-400">
      <Icon size={24} />
    </div>
    <span className="text-sm text-gray-700 dark:text-gray-300">{label}</span>
  </motion.div>
);

// Main landing page component
export default function LandingPage() {
  const [scrolled, setScrolled] = useState(false);
  const [activeSection, setActiveSection] = useState('hero');
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
  const heroRef = useRef(null);
  const gridRef = useRef(null);
  const { scrollY } = useScroll();
  
  // Transform values based on scroll position
  const headerOpacity = useTransform(scrollY, [0, 50], [0.6, 1]);
  const heroScale = useTransform(scrollY, [0, 300], [1, 0.95]);
  const heroOpacity = useTransform(scrollY, [0, 300], [1, 0.6]);
  
  // Handle scroll events
  useMotionValueEvent(scrollY, "change", (latest) => {
    setScrolled(latest > 50);
    
    // Determine active section
    if (latest < 600) {
      setActiveSection('hero');
    } else if (latest < 1200) {
      setActiveSection('about');
    } else if (latest < 1800) {
      setActiveSection('experience');
    } else {
      setActiveSection('projects');
    }
  });
  
  // Handle mouse movement for technical grid effect
  useEffect(() => {
    const handleMouseMove = (e) => {
      setMousePosition({ x: e.clientX, y: e.clientY });
    };
    
    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, []);
  
  // Animate particles in the grid based on mouse position
  useEffect(() => {
    if (!gridRef.current) return;
    
    // Create grid dots elements if not already created
    if (gridRef.current.children.length === 0) {
      for (let i = 0; i < GRID_SIZE; i++) {
        for (let j = 0; j < GRID_SIZE; j++) {
          const dot = document.createElement('div');
          dot.className = 'absolute rounded-full bg-blue-600/10 dark:bg-blue-400/10 transform transition-all duration-700';
          dot.style.width = `${DOT_SIZE}px`;
          dot.style.height = `${DOT_SIZE}px`;
          dot.style.left = `${(j * 100) / GRID_SIZE}%`;
          dot.style.top = `${(i * 100) / GRID_SIZE}%`;
          gridRef.current.appendChild(dot);
        }
      }
    }
    
    // Animate dots based on mouse position
    const dots = gridRef.current.children;
    const rect = gridRef.current.getBoundingClientRect();
    const centerX = rect.left + rect.width / 2;
    const centerY = rect.top + rect.height / 2;
    
    for (let i = 0; i < dots.length; i++) {
      const dot = dots[i];
      const dotRect = dot.getBoundingClientRect();
      const dotCenterX = dotRect.left + dotRect.width / 2;
      const dotCenterY = dotRect.top + dotRect.height / 2;
      
      // Calculate distance from mouse
      const dx = mousePosition.x - dotCenterX;
      const dy = mousePosition.y - dotCenterY;
      const distance = Math.sqrt(dx * dx + dy * dy);
      const maxDistance = 150;
      
      if (distance < maxDistance) {
        // Scale dot based on proximity to mouse
        const scale = 1 + (1 - distance / maxDistance) * 3;
        // Move dot away from mouse
        const translateX = (dx / distance) * (maxDistance - distance) * 0.15;
        const translateY = (dy / distance) * (maxDistance - distance) * 0.15;
        dot.style.transform = `translate(${-translateX}px, ${-translateY}px) scale(${scale})`;
        dot.style.backgroundColor = `rgba(59, 130, 246, ${0.1 + (1 - distance / maxDistance) * 0.2})`;
        dot.style.zIndex = "10";
      } else {
        // Reset dot
        dot.style.transform = 'translate(0, 0) scale(1)';
        dot.style.backgroundColor = '';
        dot.style.zIndex = "1";
      }
    }
  }, [mousePosition]);
  
  // Scroll to content function
  const scrollToContent = () => {
    window.scrollTo({
      top: window.innerHeight - 80,
      behavior: 'smooth'
    });
  };

  // Determine active nav link class
  const getNavClass = (section) => {
    return activeSection === section
      ? "relative py-1 font-medium text-blue-600 dark:text-blue-400 before:absolute before:bottom-0 before:left-0 before:h-0.5 before:w-full before:bg-blue-600 dark:before:bg-blue-400 before:transform before:origin-left transform transition duration-300"
      : "relative py-1 hover:text-blue-600 dark:hover:text-blue-400 before:absolute before:bottom-0 before:left-0 before:h-0.5 before:w-full before:bg-blue-600 dark:before:bg-blue-400 before:transform before:scale-x-0 before:origin-left hover:before:scale-x-100 transform transition duration-300";
  };
  
  return (
    <div className="min-h-screen bg-white text-gray-900 dark:bg-gray-900 dark:text-gray-100">
      
      {/* Hero Section with Technical Grid */}
      <motion.section 
        ref={heroRef}
        style={{
          scale: heroScale,
          opacity: heroOpacity
        }}
        className="relative min-h-screen flex items-center justify-center overflow-hidden"
      >
        <div 
          ref={gridRef}
          className="absolute inset-0 bg-gradient-to-br from-blue-50/80 to-indigo-50/80 dark:from-gray-900 dark:to-gray-900"
        ></div>
        
        <div className="absolute top-0 left-0 w-full h-20 bg-gradient-to-b from-white dark:from-gray-900"></div>
        <div className="absolute bottom-0 left-0 w-full h-20 bg-gradient-to-t from-white dark:from-gray-900"></div>
        
        <div className="container mx-auto px-6 py-12 md:py-24 relative z-10">
          <div className="grid grid-cols-1 lg:grid-cols-5 gap-8 items-center">
            <motion.div 
              initial="hidden"
              animate="visible"
              variants={staggerContainer}
              className="lg:col-span-3 text-left"
            >
              <motion.div variants={fadeInUp} className="mb-4">
                <div className="inline-flex items-center px-3 py-1 rounded-full bg-blue-100 text-blue-800 dark:bg-blue-900/50 dark:text-blue-300 text-sm font-medium mb-4">
                  <Code size={14} className="mr-1.5" /> Research Assistant at Harvard Business School
                </div>
              </motion.div>
              
              <motion.h1 
                variants={fadeInUp}
                className="text-5xl md:text-6xl lg:text-7xl font-bold mb-6 leading-tight"
              >
                <span className="bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-indigo-600 dark:from-blue-400 dark:to-indigo-400">
                  Merging Mathematics 
                </span>
                <span className="block mt-1 text-gray-800 dark:text-gray-100">
                  with Code
                </span>
              </motion.h1>
              
              <motion.p 
                variants={fadeInUp}
                className="text-lg md:text-xl text-gray-600 dark:text-gray-400 mb-8 max-w-2xl"
              >
                Computer Scientist and Applied Mathematician solving complex organizational challenges through computational solutions. Specializing in Machine Learning, AI Agents, and NLP systems.
              </motion.p>
              
              <motion.div 
                variants={fadeInUp}
                className="flex flex-wrap gap-3 mb-8"
              >
                <span className="px-4 py-2 rounded-lg text-sm font-medium bg-blue-100 text-blue-800 dark:bg-blue-900/50 dark:text-blue-300 border border-blue-200 dark:border-blue-800">
                  Machine Learning
                </span>
                <span className="px-4 py-2 rounded-lg text-sm font-medium bg-indigo-100 text-indigo-800 dark:bg-indigo-900/50 dark:text-indigo-300 border border-indigo-200 dark:border-indigo-800">
                  AI Agents
                </span>
                <span className="px-4 py-2 rounded-lg text-sm font-medium bg-purple-100 text-purple-800 dark:bg-purple-900/50 dark:text-purple-300 border border-purple-200 dark:border-purple-800">
                  NLP
                </span>
                <span className="px-4 py-2 rounded-lg text-sm font-medium bg-blue-100 text-blue-800 dark:bg-blue-900/50 dark:text-blue-300 border border-blue-200 dark:border-blue-800">
                  Computational Modeling
                </span>
              </motion.div>
              
              <motion.div 
                variants={fadeInUp}
                className="flex flex-wrap gap-4 mb-8"
              >
                <Link 
                  to="/projects" 
                  className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors flex items-center gap-2 relative overflow-hidden group shadow-md"
                >
                  <span className="z-10 relative">View Projects</span>
                  <ExternalLink size={16} className="z-10 relative" />
                  <div className="absolute inset-0 bg-blue-500 transform translate-y-full group-hover:translate-y-0 transition-transform duration-300"></div>
                </Link >
                <Link 
                  to="/documents/CV___EN.pdf" 
                  className="px-6 py-3 border-2 border-blue-600 text-blue-600 dark:text-blue-400 dark:border-blue-400 rounded-lg group hover:bg-blue-50 dark:hover:bg-blue-900/30 transition-colors flex items-center gap-2 shadow-sm"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <span>Download CV</span>
                  <svg 
                    className="w-4 h-4 transform group-hover:translate-x-1 transition-transform duration-300" 
                    viewBox="0 0 24 24" 
                    fill="none" 
                    stroke="currentColor" 
                    strokeWidth="2"
                  >
                    <path d="M12 15V3m0 12l-4-4m4 4l4-4M2 17l.621 2.485A2 2 0 0 0 4.561 21h14.878a2 2 0 0 0 1.94-1.515L22 17" />
                  </svg>
                </Link >
              </motion.div>
            </motion.div>
            
            <motion.div 
              variants={scaleUp}
              initial="hidden"
              animate="visible"
              className="lg:col-span-2"
            >
              <div className="relative">
                <div className="absolute -top-10 -left-10 w-40 h-40 bg-blue-200/30 dark:bg-blue-900/20 rounded-lg transform rotate-12 z-0"></div>
                <div className="absolute -bottom-14 -right-14 w-60 h-60 bg-indigo-200/40 dark:bg-indigo-900/20 rounded-lg transform -rotate-12 z-0"></div>
                
                <div className="relative z-10 bg-white dark:bg-gray-800 p-6 rounded-xl shadow-xl border border-gray-100 dark:border-gray-700">
                  <TypingTerminal text="const profile = {\n  name: 'Juan Lara',\n  role: 'Computer Scientist & Mathematician',\n  expertise: ['Machine Learning', 'AI Agents', 'NLP'],\n  research: 'Harvard Business School',\n  education: [\n    'B.S. Computer Science',\n    'B.S. Mathematics'\n  ],\n  passion: 'Solving complex challenges with code'\n};" />
                  
                  <div className="mt-8 grid grid-cols-3 gap-2">
                    <TechIcon icon={Cpu} label="ML Systems" delay={0.2} />
                    <TechIcon icon={Braces} label="Algorithms" delay={0.4} />
                    <TechIcon icon={Database} label="Data Analysis" delay={0.6} />
                  </div>
                </div>
              </div>
            </motion.div>
          </div>
          
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 1.8, duration: 1, repeat: Infinity, repeatType: "reverse" }}
            className="absolute bottom-8 left-1/2 transform -translate-x-1/2 cursor-pointer"
            onClick={scrollToContent}
          >
            <div className="flex flex-col items-center">
              <span className="text-sm text-gray-500 dark:text-gray-400 mb-2 font-medium">Explore More</span>
              <ChevronDown size={24} className="text-blue-600 dark:text-blue-400 animate-bounce" />
            </div>
          </motion.div>
        </div>
      </motion.section>
      
      {/* Section Previews */}
      <section className="py-20 bg-white dark:bg-gray-900">
        <div className="container mx-auto px-6 max-w-6xl">
          <motion.div 
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-100px" }}
            variants={staggerContainer}
            className="grid grid-cols-1 md:grid-cols-2 gap-10"
          >
            {/* About Preview */}
            <motion.div 
              variants={slideInLeft}
              className="bg-gradient-to-br from-gray-50 to-white dark:from-gray-800 dark:to-gray-800 p-8 rounded-xl shadow-lg hover:shadow-xl transition-all duration-500 border border-gray-100 dark:border-gray-700 transform hover:-translate-y-2"
            >
              <div className="flex items-center mb-4">
                <div className="w-10 h-10 flex items-center justify-center rounded-full bg-blue-100 text-blue-600 dark:bg-blue-900/50 dark:text-blue-300 mr-3">
                  <FileCode size={18} />
                </div>
                <h3 className="text-2xl font-bold text-gray-800 dark:text-gray-200">About Me</h3>
              </div>
              <p className="text-gray-600 dark:text-gray-300 mb-6 leading-relaxed">
                Computer scientist and mathematician focused on creating computational solutions to complex organizational challenges. Trained at Universidad Nacional de Colombia with expertise in theoretical and applied approaches.
              </p>
              <Link to="/about" className="text-blue-600 dark:text-blue-400 font-medium inline-flex items-center group">
                <span>Read more</span>
                <svg className="ml-2 w-4 h-4 transform group-hover:translate-x-1 transition-transform duration-300" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
                </svg>
              </Link>
            </motion.div>
            
            
            {/* Projects Preview */}
            <motion.div 
              variants={slideInRight}
              className="bg-gradient-to-br from-gray-50 to-white dark:from-gray-800 dark:to-gray-800 p-8 rounded-xl shadow-lg hover:shadow-xl transition-all duration-500 border border-gray-100 dark:border-gray-700 transform hover:-translate-y-2"
            >
              <div className="flex items-center mb-4">
                <div className="w-10 h-10 flex items-center justify-center rounded-full bg-blue-100 text-blue-600 dark:bg-blue-900/50 dark:text-blue-300 mr-3">
                  <Layers size={18} />
                </div>
                <h3 className="text-2xl font-bold text-gray-800 dark:text-gray-200">Projects</h3>
              </div>
              <p className="text-gray-600 dark:text-gray-300 mb-6 leading-relaxed">
                Featured projects including TextInsight library, Pharmacy Segmentation Application, and Cunservicios Platform. Each demonstrates my approach to solving real-world problems.
              </p>
              <Link to="/projects" className="text-blue-600 dark:text-blue-400 font-medium inline-flex items-center group">
                <span>Explore projects</span>
                <svg className="ml-2 w-4 h-4 transform group-hover:translate-x-1 transition-transform duration-300" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
                </svg>
              </Link>
            </motion.div>
          </motion.div>
        </div>
      </section>
      
      {/* Blog and Contact Preview */}
      <section className="py-20 bg-gray-50 dark:bg-gray-800">
        <div className="container mx-auto px-6">
          <motion.div 
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-100px" }}
            variants={staggerContainer}
            className="max-w-4xl mx-auto"
          >
            <motion.h2 
              variants={fadeInUp} 
              className="text-3xl md:text-4xl font-bold mb-6 text-center bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-indigo-600 dark:from-blue-400 dark:to-indigo-400"
            >
              Research & Writing
            </motion.h2>
            
            <motion.p 
              variants={fadeInUp}
              className="text-lg text-gray-600 dark:text-gray-300 mb-10 text-center"
            >
              Explore my thoughts on AI agents, computational organizational theory, and the intersection of mathematics and code.
            </motion.p>
            
            <motion.div 
              variants={fadeInUp}
              className="bg-white dark:bg-gray-900 p-8 rounded-xl shadow-lg border border-gray-100 dark:border-gray-700 relative overflow-hidden"
            >
              <div className="absolute top-0 right-0 w-40 h-40 bg-blue-100/50 dark:bg-blue-900/20 rounded-full -mr-20 -mt-20 z-0"></div>
              <div className="relative z-10">
                <div className="flex items-center justify-between mb-6">
                  <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200">Latest from the Blog</h3>
                  <div className="px-3 py-1 bg-blue-100 text-blue-700 dark:bg-blue-900/50 dark:text-blue-300 rounded-full text-sm font-medium">
                    New Posts Weekly
                  </div>
                </div>
                
                <div className="mb-6 pb-6 border-b border-gray-100 dark:border-gray-700">
                  <h4 className="text-lg font-medium mb-2">Multi-Agent Environments for Decision Support</h4>
                  <p className="text-gray-600 dark:text-gray-400 text-sm mb-2">
                    Exploring how multi-agent simulations can help model complex organizational dynamics.
                  </p>
                  <div className="flex items-center text-sm text-gray-500 dark:text-gray-500">
                    <span>May 10, 2025</span>
                    <span className="mx-2">•</span>
                    <span>10 min read</span>
                  </div>
                </div>
                
                <div>
                  <a 
                    href="https://blog.juanlara.dev" 
                    className="inline-flex items-center px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors gap-2 group relative overflow-hidden"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    <span className="z-10 relative">Visit Blog</span>
                    <ExternalLink size={16} className="z-10 relative transform group-hover:translate-x-1 transition-transform duration-300" />
                    <div className="absolute inset-0 bg-blue-500 transform translate-y-full group-hover:translate-y-0 transition-transform duration-300"></div>
                  </a>
                </div>
              </div>
            </motion.div>
          </motion.div>
        </div>
      </section>
      
      {/* Footer/Contact */}
      <footer className="py-12 bg-gray-900 text-white" id="contact">
        <div className="container mx-auto px-6">
          <motion.div 
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            variants={staggerContainer}
            className="max-w-4xl mx-auto"
          >
            <motion.h2 
              variants={fadeInUp} 
              className="text-2xl font-bold mb-8 text-center"
            >
              Let's Connect
            </motion.h2>
            
            <motion.div 
              variants={scaleUp}
              className="bg-gray-800 p-8 rounded-xl shadow-xl border border-gray-700 mb-8"
            >
              <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-center">
                <div>
                  <h3 className="text-xl font-semibold mb-4">Get in Touch</h3>
                  <p className="text-gray-400 mb-6">
                    I'm always interested in new projects, research collaborations, or just talking about interesting problems in AI and computational mathematics.
                  </p>
                  <div className="flex items-center mb-4">
                    <Mail className="text-blue-400 mr-3" size={18} />
                    <span className="text-gray-300">larajuand@outlook.com</span>
                  </div>
                  <div className="flex items-center">
                    <Terminal className="text-blue-400 mr-3" size={18} />
                    <span className="text-gray-300">Bogotá, Colombia</span>
                  </div>
                </div>
                
                <div>
                  <div className="flex flex-col space-y-4">
                    <a 
                      href="https://github.com/JuanLara18" 
                      target="_blank" 
                      rel="noopener noreferrer" 
                      className="flex items-center px-4 py-3 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors"
                    >
                      <Github className="mr-3 text-white" size={20} />
                      <span>Github Projects</span>
                    </a>
                    <a 
                      href="https://www.linkedin.com/in/julara/?locale=en_US" 
                      target="_blank" 
                      rel="noopener noreferrer" 
                      className="flex items-center px-4 py-3 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors"
                    >
                      <Linkedin className="mr-3 text-white" size={20} />
                      <span>LinkedIn Profile</span>
                    </a>
                    <a 
                      href="mailto:larajuand@outlook.com" 
                      className="flex items-center px-4 py-3 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors"
                    >
                      <Mail className="mr-3 text-white" size={20} />
                      <span>Send Email</span>
                    </a>
                  </div>
                </div>
              </div>
            </motion.div>
            
            <motion.p 
              variants={fadeInUp}
              className="text-center text-gray-400 text-sm"
            >
              © {new Date().getFullYear()} Juan Lara. All rights reserved.
            </motion.p>
          </motion.div>
        </div>
      </footer>
    </div>
  );
}