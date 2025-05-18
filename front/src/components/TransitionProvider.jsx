import { useState, useEffect, createContext, useContext, useRef } from 'react';
import { motion, AnimatePresence, useScroll, useSpring, useInView } from 'framer-motion';
import { useLocation } from 'react-router-dom';

// Create a context to manage transition state
const TransitionContext = createContext();

// Page transition variants with improved performance
const pageTransitions = {
  initial: {
    opacity: 0,
    y: 10
  },
  animate: {
    opacity: 1,
    y: 0,
    transition: {
      duration: 0.4,
      ease: [0.22, 1, 0.36, 1]
    }
  },
  exit: {
    opacity: 0,
    transition: {
      duration: 0.2,
      ease: [0.22, 1, 0.36, 1]
    }
  }
};

export const TransitionProvider = ({ children }) => {
  const location = useLocation();
  const [deviceInfo, setDeviceInfo] = useState({
    isMobile: false,
    prefersReducedMotion: false,
    isHighPerformance: true
  });
  
  // Performance monitoring
  const fpsCounter = useRef({ frames: 0, lastTime: performance.now() });
  const [avgFps, setAvgFps] = useState(60);
  
  // Detect device capabilities once on mount
  useEffect(() => {
    // Check device type
    const checkDevice = () => {
      const isMobile = window.innerWidth < 768 || 
                       /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
      
      // Check for reduced motion preference
      const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
      
      // Estimate device performance
      const isHighPerformance = !isMobile && !prefersReducedMotion && 
                               !(navigator.hardwareConcurrency && navigator.hardwareConcurrency < 4);
      
      setDeviceInfo({ isMobile, prefersReducedMotion, isHighPerformance });
    };
    
    checkDevice();
    
    // Setup a lightweight performance monitor
    const monitorPerformance = () => {
      fpsCounter.current.frames++;
      const now = performance.now();
      
      if (now - fpsCounter.current.lastTime > 1000) {
        const fps = Math.round(fpsCounter.current.frames * 1000 / (now - fpsCounter.current.lastTime));
        setAvgFps(prevFps => Math.round((prevFps + fps) / 2));
        
        // If FPS drops below 30, reduce animation complexity
        if (fps < 30 && deviceInfo.isHighPerformance) {
          setDeviceInfo(prev => ({ ...prev, isHighPerformance: false }));
        }
        
        fpsCounter.current = { frames: 0, lastTime: now };
      }
      
      requestAnimationFrame(monitorPerformance);
    };
    
    // Only start monitoring if not reduced motion
    if (!deviceInfo.prefersReducedMotion) {
      requestAnimationFrame(monitorPerformance);
    }
    
    // Listen for orientation or size changes
    const handleResize = () => {
      checkDevice();
    };
    
    window.addEventListener('resize', handleResize);
    window.addEventListener('orientationchange', handleResize);
    
    return () => {
      window.removeEventListener('resize', handleResize);
      window.removeEventListener('orientationchange', handleResize);
    };
  }, []);
  
  return (
    <TransitionContext.Provider value={{ ...deviceInfo, avgFps }}>
      <AnimatePresence mode="wait">
        <motion.div
          key={location.pathname}
          initial="initial"
          animate="animate"
          exit="exit"
          variants={pageTransitions}
        >
          {children}
        </motion.div>
      </AnimatePresence>
    </TransitionContext.Provider>
  );
};

// Custom hook to access transition settings
export const useTransition = () => useContext(TransitionContext);

// Improved scroll reveal component with better performance
export const ScrollReveal = ({ 
  children,
  threshold = 0.1,
  direction = "up",
  distance = 30,
  delay = 0,
  duration = 0.7,
  once = true,
  className = "",
  ...props 
}) => {
  const { isMobile, prefersReducedMotion, isHighPerformance } = useTransition();
  const ref = useRef(null);
  const isInView = useInView(ref, { 
    once, 
    margin: `-${threshold * 100}px 0px` 
  });
  
  // Adjust animation parameters based on device capability
  const adjustedDistance = isMobile ? Math.min(distance, 20) : distance;
  const adjustedDuration = isMobile ? Math.min(duration, 0.5) : duration;
  
  // Simpler animation for reduced motion or low performance
  const shouldUseSimpleAnimation = prefersReducedMotion || !isHighPerformance;
  
  // Calculate motion properties based on direction
  const getMotionProps = () => {
    if (shouldUseSimpleAnimation) {
      return { opacity: isInView ? 1 : 0 };
    }
    
    const directionProps = {
      y: direction === "up" ? (isInView ? 0 : adjustedDistance) : 
         direction === "down" ? (isInView ? 0 : -adjustedDistance) : 0,
      x: direction === "left" ? (isInView ? 0 : adjustedDistance) : 
         direction === "right" ? (isInView ? 0 : -adjustedDistance) : 0
    };
    
    return {
      opacity: isInView ? 1 : 0,
      ...directionProps,
      transition: {
        duration: adjustedDuration,
        delay: isMobile ? delay * 0.5 : delay,
        ease: [0.22, 1, 0.36, 1]
      }
    };
  };
  
  return (
    <motion.div
      ref={ref}
      className={className}
      animate={getMotionProps()}
      {...props}
    >
      {children}
    </motion.div>
  );
};

// Parallax scroll component for subtle depth effects
export const ParallaxScroll = ({ children, speed = 0.2, className = "", ...props }) => {
  const { isMobile, prefersReducedMotion, isHighPerformance } = useTransition();
  const ref = useRef(null);
  const { scrollYProgress } = useScroll({
    target: ref,
    offset: ["start end", "end start"]
  });
  
  // Skip parallax if reduced motion or low performance
  if (prefersReducedMotion || !isHighPerformance) {
    return <div ref={ref} className={className} {...props}>{children}</div>;
  }
  
  // Adjust parallax effect based on device
  const adjustedSpeed = isMobile ? speed * 0.5 : speed;
  
  // Use spring physics for smoother parallax
  const y = useSpring(
    useTransform(scrollYProgress, [0, 1], [0, adjustedSpeed * 100]), 
    { stiffness: 100, damping: 30, restDelta: 0.001 }
  );
  
  return (
    <motion.div
      ref={ref}
      style={{ y }}
      className={className}
      {...props}
    >
      {children}
    </motion.div>
  );
};

// Optimized stagger container with adaptive timing
export const StaggerContainer = ({ 
  children, 
  staggerDelay = 0.1, 
  childVariants,
  className = "", 
  ...props 
}) => {
  const { isMobile, prefersReducedMotion } = useTransition();
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, amount: 0.2 });
  
  // Default variants if none provided
  const defaultVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { 
      opacity: 1, 
      y: 0,
      transition: { duration: 0.5 }
    }
  };
  
  // Use provided variants or default
  const variants = childVariants || defaultVariants;
  
  // Adjust stagger timing based on device
  const adjustedDelay = prefersReducedMotion ? 0 : (isMobile ? staggerDelay * 0.5 : staggerDelay);
  
  return (
    <motion.div
      ref={ref}
      className={className}
      initial="hidden"
      animate={isInView ? "visible" : "hidden"}
      variants={{
        hidden: {},
        visible: {
          transition: {
            staggerChildren: adjustedDelay,
            delayChildren: 0.1
          }
        }
      }}
      {...props}
    >
      {React.Children.map(children, child => {
        if (React.isValidElement(child)) {
          return React.cloneElement(child, {
            variants: child.props.variants || variants
          });
        }
        return child;
      })}
    </motion.div>
  );
};

// Improved hover effect with adaptive motion
export const HoverMotion = ({ 
  children, 
  scale = 1.03, 
  y = -3, 
  duration = 0.3,
  className = "", 
  ...props 
}) => {
  const { isMobile, prefersReducedMotion } = useTransition();
  
  // Skip hover animations on mobile or reduced motion
  if (isMobile || prefersReducedMotion) {
    return <div className={className} {...props}>{children}</div>;
  }
  
  return (
    <motion.div
      className={className}
      whileHover={{ 
        scale, 
        y,
        transition: { 
          duration, 
          ease: [0.22, 1, 0.36, 1]
        }
      }}
      {...props}
    >
      {children}
    </motion.div>
  );
};