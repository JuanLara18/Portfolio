// Centralized motion variants and helpers for consistent transitions
import { useMemo } from 'react';

// Easing
export const easeStandard = [0.22, 1, 0.36, 1];

// Basic fades and slides
export const variants = {
  fadeInUp: (duration = 0.7, distance = 30, delay = 0) => ({
    hidden: { opacity: 0, y: distance },
    visible: { opacity: 1, y: 0, transition: { duration, delay, ease: easeStandard } }
  }),
  fadeInDown: (duration = 0.7, distance = 30, delay = 0) => ({
    hidden: { opacity: 0, y: -distance },
    visible: { opacity: 1, y: 0, transition: { duration, delay, ease: easeStandard } }
  }),
  fadeInLeft: (duration = 0.7, distance = 40, delay = 0) => ({
    hidden: { opacity: 0, x: distance },
    visible: { opacity: 1, x: 0, transition: { duration, delay, ease: easeStandard } }
  }),
  fadeInRight: (duration = 0.7, distance = 40, delay = 0) => ({
    hidden: { opacity: 0, x: -distance },
    visible: { opacity: 1, x: 0, transition: { duration, delay, ease: easeStandard } }
  }),
  scaleUp: (duration = 0.6, delay = 0) => ({
    hidden: { opacity: 0, scale: 0.9 },
    visible: { opacity: 1, scale: 1, transition: { duration, delay, ease: easeStandard } }
  }),
  simpleFade: (duration = 0.3, delay = 0) => ({
    hidden: { opacity: 0 },
    visible: { opacity: 1, transition: { duration, delay, ease: easeStandard } }
  }),
  // Page transitions
  page: {
    initial: { opacity: 0, y: 10 },
    animate: { opacity: 1, y: 0, transition: { duration: 0.4, ease: easeStandard } },
    exit: { opacity: 0, transition: { duration: 0.2, ease: easeStandard } }
  },
  // Stagger container
  stagger: (staggerChildren = 0.15, delayChildren = 0.2) => ({
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: { staggerChildren, delayChildren }
    }
  })
};

// Hook for ScrollReveal variants to keep stability
export const useScrollRevealVariants = ({ direction = 'up', distance = 30, duration = 0.7, delay = 0, simple = false }) =>
  useMemo(() => {
    if (simple) return variants.simpleFade(duration, delay);
    const initial = { opacity: 0 };
    if (direction === 'up') initial.y = distance;
    if (direction === 'down') initial.y = -distance;
    if (direction === 'left') initial.x = distance;
    if (direction === 'right') initial.x = -distance;
    return {
      hidden: initial,
      visible: { opacity: 1, x: 0, y: 0, transition: { duration, delay, ease: easeStandard } }
    };
  }, [direction, distance, duration, delay, simple]);
