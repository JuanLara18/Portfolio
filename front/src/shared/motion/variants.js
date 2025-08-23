// Centralized motion variants and helpers for consistent transitions
import { useMemo } from 'react';

// Improved easing for more professional feel
export const easeStandard = [0.25, 0.1, 0.25, 1.0];
export const easeSmooth = [0.4, 0.0, 0.2, 1.0];
export const easeGentle = [0.25, 0.46, 0.45, 0.94];

// Professional scroll-based variants with bidirectional support
export const variants = {
  // Enhanced fade animations with better easing
  fadeInUp: (duration = 0.8, distance = 40, delay = 0) => ({
    hidden: { opacity: 0, y: distance },
    visible: { 
      opacity: 1, 
      y: 0, 
      transition: { 
        duration, 
        delay, 
        ease: easeGentle,
        type: "tween"
      } 
    }
  }),
  fadeInDown: (duration = 0.8, distance = 40, delay = 0) => ({
    hidden: { opacity: 0, y: -distance },
    visible: { 
      opacity: 1, 
      y: 0, 
      transition: { 
        duration, 
        delay, 
        ease: easeGentle,
        type: "tween"
      } 
    }
  }),
  fadeInLeft: (duration = 0.8, distance = 50, delay = 0) => ({
    hidden: { opacity: 0, x: distance },
    visible: { 
      opacity: 1, 
      x: 0, 
      transition: { 
        duration, 
        delay, 
        ease: easeGentle,
        type: "tween"
      } 
    }
  }),
  fadeInRight: (duration = 0.8, distance = 50, delay = 0) => ({
    hidden: { opacity: 0, x: -distance },
    visible: { 
      opacity: 1, 
      x: 0, 
      transition: { 
        duration, 
        delay, 
        ease: easeGentle,
        type: "tween"
      } 
    }
  }),
  // Smooth scale with professional timing
  scaleUp: (duration = 0.7, delay = 0) => ({
    hidden: { opacity: 0, scale: 0.92 },
    visible: { 
      opacity: 1, 
      scale: 1, 
      transition: { 
        duration, 
        delay, 
        ease: easeGentle,
        type: "tween"
      } 
    }
  }),
  // Subtle fade for smooth transitions
  simpleFade: (duration = 0.5, delay = 0) => ({
    hidden: { opacity: 0 },
    visible: { 
      opacity: 1, 
      transition: { 
        duration, 
        delay, 
        ease: easeSmooth 
      } 
    }
  }),
  // Enhanced page transitions
  page: {
    initial: { opacity: 0, y: 8 },
    animate: { 
      opacity: 1, 
      y: 0, 
      transition: { 
        duration: 0.6, 
        ease: easeGentle 
      } 
    },
    exit: { 
      opacity: 0, 
      transition: { 
        duration: 0.3, 
        ease: easeSmooth 
      } 
    }
  },
  // Professional stagger with better timing
  stagger: (staggerChildren = 0.12, delayChildren = 0.15) => ({
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: { 
        staggerChildren, 
        delayChildren,
        ease: easeGentle
      }
    }
  }),
  // Scroll-triggered variants for professional reveal
  scrollReveal: {
    up: (duration = 0.8, distance = 35) => ({
      hidden: { opacity: 0, y: distance },
      visible: { 
        opacity: 1, 
        y: 0,
        transition: { 
          duration, 
          ease: easeGentle,
          type: "tween"
        }
      }
    }),
    down: (duration = 0.8, distance = 35) => ({
      hidden: { opacity: 0, y: -distance },
      visible: { 
        opacity: 1, 
        y: 0,
        transition: { 
          duration, 
          ease: easeGentle,
          type: "tween"
        }
      }
    }),
    left: (duration = 0.8, distance = 45) => ({
      hidden: { opacity: 0, x: distance },
      visible: { 
        opacity: 1, 
        x: 0,
        transition: { 
          duration, 
          ease: easeGentle,
          type: "tween"
        }
      }
    }),
    right: (duration = 0.8, distance = 45) => ({
      hidden: { opacity: 0, x: -distance },
      visible: { 
        opacity: 1, 
        x: 0,
        transition: { 
          duration, 
          ease: easeGentle,
          type: "tween"
        }
      }
    }),
    scale: (duration = 0.7) => ({
      hidden: { opacity: 0, scale: 0.93 },
      visible: { 
        opacity: 1, 
        scale: 1,
        transition: { 
          duration, 
          ease: easeGentle,
          type: "tween"
        }
      }
    })
  }
};

// Hook for ScrollReveal variants to keep stability and add professional viewport settings
export const useScrollRevealVariants = ({ 
  direction = 'up', 
  distance = 35, 
  duration = 0.8, 
  delay = 0, 
  simple = false 
}) =>
  useMemo(() => {
    if (simple) return variants.simpleFade(duration, delay);
    
    return variants.scrollReveal[direction] 
      ? variants.scrollReveal[direction](duration, distance)
      : variants.scrollReveal.up(duration, distance);
  }, [direction, distance, duration, delay, simple]);

// Professional viewport settings for consistent scroll reveals
export const defaultViewportSettings = {
  once: true,
  margin: "-50px 0px -50px 0px", // More generous trigger area
  amount: 0.1 // Trigger when 10% is visible
};

// Smooth viewport settings for earlier triggers
export const earlyViewportSettings = {
  once: true,
  margin: "-20px 0px -20px 0px",
  amount: 0.05
};
