import { useEffect, useRef, useState } from 'react';

/**
 * Professional scroll reveal hook for bidirectional animations
 * Provides smooth, professional animations that work both ways
 */
export const useScrollReveal = ({
  threshold = 0.1,
  rootMargin = '-50px 0px -50px 0px',
  triggerOnce = false
} = {}) => {
  const [isVisible, setIsVisible] = useState(false);
  const [hasBeenVisible, setHasBeenVisible] = useState(false);
  const elementRef = useRef(null);

  useEffect(() => {
    const element = elementRef.current;
    if (!element) return;

    const observer = new IntersectionObserver(
      ([entry]) => {
        const isIntersecting = entry.isIntersecting;
        
        if (isIntersecting) {
          setIsVisible(true);
          setHasBeenVisible(true);
        } else {
          // Allow re-triggering animations unless triggerOnce is true
          if (!triggerOnce || !hasBeenVisible) {
            setIsVisible(false);
          }
        }
      },
      {
        threshold,
        rootMargin,
      }
    );

    observer.observe(element);

    return () => {
      observer.unobserve(element);
    };
  }, [threshold, rootMargin, triggerOnce, hasBeenVisible]);

  return [elementRef, isVisible];
};

/**
 * Enhanced scroll reveal hook with direction detection
 * Provides different animations based on scroll direction
 */
export const useDirectionalScrollReveal = (options = {}) => {
  const [ref, isVisible] = useScrollReveal(options);
  const [scrollDirection, setScrollDirection] = useState('down');
  const lastScrollY = useRef(0);

  useEffect(() => {
    const handleScroll = () => {
      const currentScrollY = window.scrollY;
      const direction = currentScrollY > lastScrollY.current ? 'down' : 'up';
      
      if (Math.abs(currentScrollY - lastScrollY.current) > 10) {
        setScrollDirection(direction);
        lastScrollY.current = currentScrollY;
      }
    };

    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return [ref, isVisible, scrollDirection];
};
