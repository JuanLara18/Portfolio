import { useState, useEffect, useRef } from 'react';
import { motion, useScroll, useTransform, useMotionValueEvent } from 'framer-motion';
import { variants as motionVariants } from '../shared/motion';
import { ChevronDown, Github, Linkedin, Mail, ExternalLink, Code, Terminal, Database, Server, Cpu, TerminalSquare, BrainCircuit, FileCode, Braces, Layers } from 'lucide-react';
import { Link } from 'react-router-dom';

// Import enhanced components via UI barrel
import { TypingTerminal, TechIcon, ParticleBackground } from '../components/ui';
import { HoverMotion } from '../components/layout/TransitionProvider';

// Animation variants
const fadeInUp = motionVariants.fadeInUp();
const staggerContainer = motionVariants.stagger();
const slideInRight = motionVariants.fadeInRight();
const slideInLeft = motionVariants.fadeInLeft();
const scaleUp = motionVariants.scaleUp();

// Main landing page component
export default function LandingPage() {
  const [scrolled, setScrolled] = useState(false);
  const [activeSection, setActiveSection] = useState('hero');
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
  const heroRef = useRef(null);
  const titleRef = useRef(null);
  const { scrollY } = useScroll();
  
  // Enhanced transform values based on scroll position for parallax effects
  const heroOpacity = useTransform(scrollY, [150, 500], [1, 0.9]);
  const heroScale = useTransform(scrollY, [150, 500], [1, 0.98]);
  const heroY = useTransform(scrollY, [150, 500], [0, 10]);
  const headingY = useTransform(scrollY, [0, 300], [0, -15]);
  
  // Parallax for decorative elements
  const bgElement1Y = useTransform(scrollY, [0, 300], [0, 30]);
  const bgElement2Y = useTransform(scrollY, [0, 300], [0, -20]);
  
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
  
  // Handle mouse movement with debounce for better performance
  useEffect(() => {
    let timeoutId;
    
    const handleMouseMove = (e) => {
      clearTimeout(timeoutId);
      
      timeoutId = setTimeout(() => {
        setMousePosition({ x: e.clientX, y: e.clientY });
      }, 10); // Slight debounce for smoother experience
    };
    
    window.addEventListener('mousemove', handleMouseMove);
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      clearTimeout(timeoutId);
    };
  }, []);
  
  // Enhanced scroll to content function with smooth acceleration
  const scrollToContent = () => {
    const targetPosition = window.innerHeight - 80;
    const startPosition = window.scrollY;
    const distance = targetPosition - startPosition;
    const duration = 1000;
    let start = null;
    
    // Easing function for smooth acceleration/deceleration
    const easeInOutCubic = (t) => {
      return t < 0.5
        ? 4 * t * t * t
        : 1 - Math.pow(-2 * t + 2, 3) / 2;
    };
    
    const animate = (timestamp) => {
      if (!start) start = timestamp;
      const elapsed = timestamp - start;
      const progress = Math.min(elapsed / duration, 1);
      const eased = easeInOutCubic(progress);
      
      window.scrollTo(0, startPosition + distance * eased);
      
      if (progress < 1) {
        window.requestAnimationFrame(animate);
      }
    };
    
    window.requestAnimationFrame(animate);
  };

  // Determine active nav link class
  const getNavClass = (section) => {
    return activeSection === section
      ? "relative py-1 font-medium text-blue-600 dark:text-blue-400 before:absolute before:bottom-0 before:left-0 before:h-0.5 before:w-full before:bg-blue-600 dark:before:bg-blue-400 before:transform before:origin-left transform transition duration-300"
      : "relative py-1 hover:text-blue-600 dark:hover:text-blue-400 before:absolute before:bottom-0 before:left-0 before:h-0.5 before:w-full before:bg-blue-600 dark:before:bg-blue-400 before:transform before:scale-x-0 before:origin-left hover:before:scale-x-100 transform transition duration-300";
  };
  
  return (
    <div className="min-h-screen bg-white text-gray-900 dark:bg-gray-900 dark:text-gray-100">
      
      {/* Hero Section with Enhanced Technical Grid */}
      <motion.section 
        ref={heroRef}
        style={{
          scale: heroScale,
          opacity: heroOpacity,
          y: heroY // Apply parallax effect on scroll
        }}
        className="relative min-h-screen flex items-center justify-center overflow-hidden"
      >
        {/* Enhanced Particle Grid Background */}
  <ParticleBackground mousePosition={mousePosition} />
        
        {/* Floating decorative elements with parallax effect */}
        <motion.div 
          className="absolute top-20 right-10 w-96 h-96 rounded-full bg-blue-200/20 dark:bg-blue-900/10 blur-3xl -z-10"
          style={{ y: bgElement1Y }}
        />
        <motion.div 
          className="absolute bottom-40 left-10 w-64 h-64 rounded-full bg-indigo-200/30 dark:bg-indigo-900/10 blur-3xl -z-10"
          style={{ y: bgElement2Y }}
        />
        
        {/* Gradient overlays for smooth fading */}
        <div className="absolute top-0 left-0 w-full h-32 bg-gradient-to-b from-white dark:from-gray-900 to-transparent z-0"></div>
        <div className="absolute bottom-0 left-0 w-full h-32 bg-gradient-to-t from-white dark:from-gray-900 to-transparent z-0"></div>
        
        <div className="container mx-auto px-6 py-12 md:py-24 relative z-10">
          <div className="grid grid-cols-1 lg:grid-cols-5 gap-8 items-center">
            <motion.div 
              initial="hidden"
              animate="visible"
              variants={staggerContainer}
              className="lg:col-span-3 text-left"
              style={{ y: headingY }} // Counter-parallax for content
            >
              <motion.div variants={fadeInUp} className="mb-4">
                <div className="inline-flex items-center px-3 py-1 rounded-full bg-blue-100 text-blue-800 dark:bg-blue-900/50 dark:text-blue-300 text-sm font-medium mb-4 backdrop-blur-sm">
                  <Code size={14} className="mr-1.5" /> LLM/ML Specialist at GenomAI
                </div>
              </motion.div>
              
              {/* Enhanced text reveal animation */}
              <motion.h1 
                ref={titleRef}
                variants={fadeInUp}
                className="text-5xl md:text-6xl lg:text-7xl font-bold mb-6 leading-normal"
              >
                <motion.span 
                  className="bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-indigo-600 dark:from-blue-400 dark:to-indigo-400 inline-block  pb-1"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.8, delay: 0.2 }}
                >
                  Transforming Numbers
                </motion.span>
                <motion.span 
                  className="block mt-1 text-gray-800 dark:text-gray-100"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.8, delay: 0.4 }}
                >
                  into Action
                </motion.span>
              </motion.h1>
              
              <motion.p 
                variants={fadeInUp}
                className="text-lg md:text-xl text-gray-600 dark:text-gray-400 mb-8 max-w-2xl leading-relaxed"
              >
                I build end-to-end AI systems that transform complex requirements into scalable, production-ready solutions. Specializing in LLM fine-tuning, RAG architectures, and cloud deployment.
              </motion.p>
              
              <motion.div 
                variants={fadeInUp}
                className="flex flex-wrap gap-3 mb-8"
              >
                <HoverMotion as={motion.span} extraWhileHover={{ boxShadow: "0 10px 25px -5px rgba(59, 130, 246, 0.5)" }}
                  className="px-4 py-2 rounded-lg text-sm font-medium bg-blue-100 text-blue-800 dark:bg-blue-900/50 dark:text-blue-300 border border-blue-200 dark:border-blue-800 backdrop-blur-sm transition-all duration-300"
                >
                  LLM & RAG Systems
                </HoverMotion>
                <HoverMotion as={motion.span} extraWhileHover={{ boxShadow: "0 10px 25px -5px rgba(67, 56, 202, 0.5)" }}
                  className="px-4 py-2 rounded-lg text-sm font-medium bg-indigo-100 text-indigo-800 dark:bg-indigo-900/50 dark:text-indigo-300 border border-indigo-200 dark:border-indigo-800 backdrop-blur-sm transition-all duration-300"
                >
                  Generative AI
                </HoverMotion>
                <HoverMotion as={motion.span} extraWhileHover={{ boxShadow: "0 10px 25px -5px rgba(147, 51, 234, 0.5)" }}
                  className="px-4 py-2 rounded-lg text-sm font-medium bg-purple-100 text-purple-800 dark:bg-purple-900/50 dark:text-purple-300 border border-purple-200 dark:border-purple-800 backdrop-blur-sm transition-all duration-300"
                >
                  NLP
                </HoverMotion>
                <HoverMotion as={motion.span} extraWhileHover={{ boxShadow: "0 10px 25px -5px rgba(59, 130, 246, 0.5)" }}
                  className="px-4 py-2 rounded-lg text-sm font-medium bg-blue-100 text-blue-800 dark:bg-blue-900/50 dark:text-blue-300 border border-blue-200 dark:border-blue-800 backdrop-blur-sm transition-all duration-300"
                >
                  Cloud Deployment
                </HoverMotion>
              </motion.div>
              
              {/* Enhanced CTA buttons */}
              <motion.div 
                variants={fadeInUp}
                className="flex flex-wrap gap-4 mb-8"
              >
                <Link 
                  to="/projects" 
                  className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-all duration-300 flex items-center gap-2 relative overflow-hidden group shadow-lg"
                >
                  <span className="z-10 relative">View Projects</span>
                  <ExternalLink size={16} className="z-10 relative transition-transform duration-300 group-hover:translate-x-1" />
                  <div className="absolute inset-0 bg-blue-500 transform translate-y-full group-hover:translate-y-0 transition-transform duration-500"></div>
                  {/* Add subtle light reflection effect */}
                  <motion.div 
                    className="absolute inset-0 opacity-0 group-hover:opacity-30 bg-gradient-to-r from-transparent via-white to-transparent skew-x-20 transition-opacity duration-1000"
                    animate={{ 
                      x: ["200%", "-200%"],
                      transition: { 
                        repeat: Infinity, 
                        repeatType: "loop", 
                        duration: 2.5,
                        ease: "easeInOut",
                        repeatDelay: 0.5
                      } 
                    }}
                  />
                </Link>
                <Link 
                  to="/documents/CV___EN.pdf" 
                  className="px-6 py-3 border-2 border-blue-600 text-blue-600 dark:text-blue-400 dark:border-blue-400 rounded-lg group hover:bg-blue-50 dark:hover:bg-blue-900/30 transition-all duration-300 flex items-center gap-2 shadow-sm relative overflow-hidden"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <span className="relative z-10">Download CV</span>
                  <svg 
                    className="w-4 h-4 transform transition-transform duration-300 group-hover:translate-x-1 group-hover:translate-y-1 relative z-10" 
                    viewBox="0 0 24 24" 
                    fill="none" 
                    stroke="currentColor" 
                    strokeWidth="2"
                  >
                    <path d="M12 15V3m0 12l-4-4m4 4l4-4M2 17l.621 2.485A2 2 0 0 0 4.561 21h14.878a2 2 0 0 0 1.94-1.515L22 17" />
                  </svg>
                  {/* Add subtle glow on hover */}
                  <motion.div 
                    className="absolute inset-0 opacity-0 group-hover:opacity-20 bg-blue-300 dark:bg-blue-700 rounded-lg transition-opacity duration-300"
                  />
                </Link>
              </motion.div>
            </motion.div>
            
            {/* Enhanced Terminal Card */}
            <HoverMotion as={motion.div}
              variants={scaleUp}
              initial="hidden"
              animate="visible"
              className="lg:col-span-2"
              y={-5}
              duration={0.3}
            >
              <div className="relative">
                {/* Animated decorative shapes */}
                <motion.div 
                  className="absolute -top-10 -left-10 w-40 h-40 bg-blue-200/30 dark:bg-blue-900/20 rounded-lg z-0"
                  initial={{ rotate: 12, scale: 0.9 }}
                  animate={{ 
                    rotate: [12, 15, 12, 9, 12],
                    scale: [0.9, 1, 0.95, 1, 0.9],
                    transition: { 
                      duration: 10, 
                      repeat: Infinity,
                      repeatType: "reverse"
                    }
                  }}
                />
                <motion.div 
                  className="absolute -bottom-14 -right-14 w-60 h-60 bg-indigo-200/40 dark:bg-indigo-900/20 rounded-lg z-0"
                  initial={{ rotate: -12, scale: 0.95 }}
                  animate={{ 
                    rotate: [-12, -9, -12, -15, -12],
                    scale: [0.95, 1.05, 1, 0.98, 0.95],
                    transition: { 
                      duration: 12, 
                      repeat: Infinity,
                      repeatType: "reverse"
                    }
                  }}
                />
                
                {/* Enhanced terminal card with improved shadow */}
                <div className="relative z-10 bg-white dark:bg-gray-800 p-6 rounded-xl shadow-[0_10px_40px_-15px_rgba(0,0,0,0.3)] dark:shadow-[0_10px_40px_-15px_rgba(0,0,0,0.7)] border border-gray-100 dark:border-gray-700 transition-all duration-500">
                  <TypingTerminal text="const profile = {\n  name: 'Juan Lara',\n  role: 'LLM/ML Specialist @ GenomAI',\n  expertise: ['RAG Systems', 'LLM Fine-tuning', 'Vector DBs'],\n  stack: ['Python', 'PyTorch', 'LangChain', 'GCP'],\n  education: [\n    'B.S. Computer Science',\n    'B.S. Mathematics'\n  ],\n  mission: 'Transform complex AI requirements into scalable solutions'\n};" />

                  <div className="mt-8 grid grid-cols-3 gap-2">
                    <TechIcon icon={BrainCircuit} label="LLM Engineering" delay={0.2} />
                    <TechIcon icon={Database} label="RAG Systems" delay={0.4} />
                    <TechIcon icon={Server} label="Cloud Deployment" delay={0.6} />
                  </div>
                </div>
              </div>
            </HoverMotion>
          </div>
          
          {/* Enhanced scroll indicator */}
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ 
              delay: 1.8, 
              duration: 0.8
            }}
            className="absolute bottom-8 left-1/2 transform -translate-x-1/2 cursor-pointer"
            onClick={scrollToContent}
          >
            <div className="flex flex-col items-center">
              <span className="text-sm text-gray-500 dark:text-gray-400 mb-2 font-medium">Explore More</span>
              <motion.div
                animate={{ 
                  y: [0, 8, 0],
                  transition: { 
                    duration: 1.5, 
                    repeat: Infinity,
                    repeatType: "loop"
                  }
                }}
              >
                <ChevronDown size={24} className="text-blue-600 dark:text-blue-400" />
              </motion.div>
            </div>
          </motion.div>
          
        {/* Subtle cursor effect for desktop only */}
        <motion.div 
          className="fixed w-6 h-6 rounded-full border-2 border-blue-400/30 dark:border-blue-500/30 pointer-events-none z-50 hidden md:block"
          style={{ 
            left: mousePosition.x - 12, 
            top: mousePosition.y - 12,
            opacity: scrolled ? 0 : 0.6
          }}
          animate={{ 
            scale: [1, 1.2, 1],
            transition: { duration: 1, repeat: Infinity }
          }}
        />
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
                I'm a Computer Scientist & Mathematician with 3+ years developing production-ready generative AI solutions. Currently building AI-powered clinical decision support systems at GenomAI, with previous research experience at Harvard University.
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
                From TextInsight (multilingual NLP library) to scalable RAG systems, I build end-to-end AI solutions that bridge advanced ML techniques with robust engineering practices. Each project emphasizes production-ready deployment and measurable business impact.
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
              Coming soon: reflections on machine intelligence, computational organization theory, and the logic that connects math with code.
            </motion.p>
            
            <motion.div 
              variants={fadeInUp}
              className="bg-white dark:bg-gray-900 p-8 rounded-xl shadow-lg border border-gray-100 dark:border-gray-700 relative overflow-hidden"
              whileHover={{ y: -5, transition: { duration: 0.3 } }}
            >
              <div className="absolute top-0 right-0 w-40 h-40 bg-blue-100/50 dark:bg-blue-900/20 rounded-full -mr-20 -mt-20 z-0"></div>
              <div className="relative z-10">
                <div className="flex items-center justify-between mb-6">
                  <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200">Latest from the Blog</h3>
                  <div className="px-3 py-1 bg-blue-100 text-blue-700 dark:bg-blue-900/50 dark:text-blue-300 rounded-full text-sm font-medium">
                    Actively posting
                  </div>
                </div>
                
                {/* <div className="mb-6 pb-6 border-b border-gray-100 dark:border-gray-700">
                  <h4 className="text-lg font-medium mb-2">Multi-Agent Environments for Decision Support</h4>
                  <p className="text-gray-600 dark:text-gray-400 text-sm mb-2">
                    Exploring how multi-agent simulations can help model complex organizational dynamics.
                  </p>
                  <div className="flex items-center text-sm text-gray-500 dark:text-gray-500">
                    <span>May 10, 2025</span>
                    <span className="mx-2">•</span>
                    <span>10 min read</span>
                  </div>
                </div> */}
                
                <div>
                  <Link
                    to="/blog"
                    className="flex items-center justify-center px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-all duration-300 relative overflow-hidden group shadow-lg"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    <span className="z-10 relative">Visit Blog</span>
                    <ExternalLink size={16} className="z-10 relative transform group-hover:translate-x-1 transition-transform duration-300" />
                    <div className="absolute inset-0 bg-blue-500 transform translate-y-full group-hover:translate-y-0 transition-transform duration-300"></div>
                    {/* Light reflection effect */}
                    <motion.div
                      className="absolute inset-0 opacity-0 group-hover:opacity-30 bg-gradient-to-r from-transparent via-white to-transparent skew-x-20 transition-opacity duration-1000"
                      animate={{ 
                        x: ["200%", "-200%"],
                        transition: { 
                          repeat: Infinity, 
                          repeatType: "loop", 
                          duration: 2.5,
                          ease: "easeInOut",
                          repeatDelay: 0.5
                        } 
                      }}
                    />
                  </Link>
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
            
            <HoverMotion as={motion.div}
              className="bg-gray-800 p-8 rounded-xl shadow-xl border border-gray-700 mb-8"
              y={-5}
              duration={0.3}
              variants={scaleUp}
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
                    <HoverMotion as={motion.a}
                      href="https://github.com/JuanLara18" 
                      target="_blank" 
                      rel="noopener noreferrer" 
                      className="flex items-center px-4 py-3 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors relative overflow-hidden group"
                      y={-2}
                      duration={0.2}
                    >
                      <Github className="mr-3 text-white" size={20} />
                      <span>Github Projects</span>
                      <motion.div 
                        className="absolute inset-0 opacity-0 group-hover:opacity-10 bg-gradient-to-r from-transparent via-white to-transparent skew-x-20"
                        animate={{ 
                          x: ["200%", "-200%"],
                          transition: { 
                            repeat: Infinity, 
                            repeatType: "loop", 
                            duration: 2,
                            ease: "easeInOut",
                            repeatDelay: 0.2
                          } 
                        }}
                      />
                    </HoverMotion>
                    <HoverMotion as={motion.a}
                      href="https://www.linkedin.com/in/julara/?locale=en_US" 
                      target="_blank" 
                      rel="noopener noreferrer" 
                      className="flex items-center px-4 py-3 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors relative overflow-hidden group"
                      y={-2}
                      duration={0.2}
                    >
                      <Linkedin className="mr-3 text-white" size={20} />
                      <span>LinkedIn Profile</span>
                      <motion.div 
                        className="absolute inset-0 opacity-0 group-hover:opacity-10 bg-gradient-to-r from-transparent via-white to-transparent skew-x-20"
                        animate={{ 
                          x: ["200%", "-200%"],
                          transition: { 
                            repeat: Infinity, 
                            repeatType: "loop", 
                            duration: 2,
                            ease: "easeInOut",
                            repeatDelay: 0.2
                          } 
                        }}
                      />
                    </HoverMotion>
                    <HoverMotion as={motion.a}
                      href="mailto:larajuand@outlook.com" 
                      className="flex items-center px-4 py-3 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors relative overflow-hidden group"
                      y={-2}
                      duration={0.2}
                    >
                      <Mail className="mr-3 text-white" size={20} />
                      <span>Send Email</span>
                      <motion.div 
                        className="absolute inset-0 opacity-0 group-hover:opacity-10 bg-gradient-to-r from-transparent via-white to-transparent skew-x-20"
                        animate={{ 
                          x: ["200%", "-200%"],
                          transition: { 
                            repeat: Infinity, 
                            repeatType: "loop", 
                            duration: 2,
                            ease: "easeInOut",
                            repeatDelay: 0.2
                          } 
                        }}
                      />
                    </HoverMotion>
                  </div>
                </div>
              </div>
            </HoverMotion>
            
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