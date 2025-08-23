import React from 'react';
import { motion } from 'framer-motion';
import { useTransition } from '../layout/TransitionProvider/TransitionProvider';
import { easeStandard } from '../../utils';

// Compute default hover style based on type
const getHoverStyle = (type) => {
  switch (type) {
    case 'scale':
      return { scale: 1.02, transition: { duration: 0.3, ease: easeStandard } };
    case 'liftScale':
      return { y: -8, scale: 1.02, transition: { duration: 0.3, ease: easeStandard } };
    case 'lift':
    default:
      return { y: -5, transition: { duration: 0.3, ease: easeStandard } };
  }
};

export const MotionCard = ({
  as: Component = motion.article,
  className = '',
  hover = 'lift', // 'lift' | 'scale' | 'liftScale' | 'none'
  whileHover: customWhileHover,
  children,
  ...rest
}) => {
  const { isMobile, prefersReducedMotion } = useTransition();
  const disableHover = isMobile || prefersReducedMotion || hover === 'none';
  const hoverStyle = customWhileHover || getHoverStyle(hover);

  return (
    <Component
      className={className}
      whileHover={disableHover ? undefined : hoverStyle}
      {...rest}
    >
      {children}
    </Component>
  );
};

export default MotionCard;
