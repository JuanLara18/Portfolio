import { useEffect, useRef } from 'react';

// Simplified particle animation constants
const GRID_SIZE = 24; // Reduced from 36 to 20 (64% fewer particles)
const DOT_SIZE = 5; // Slightly larger to compensate for fewer particles
const MAX_DISTANCE = 150;
const UPDATE_INTERVAL = 25; // Only update every 50ms instead of every frame

const ParticleGridBackground = ({ mousePosition }) => {
  const gridRef = useRef(null);
  
  // Create and animate particles on mount and when mouse moves
  useEffect(() => {
    if (!gridRef.current) return;
    
    // Create grid dots if not already created
    if (gridRef.current.children.length === 0) {
      // Create a document fragment to batch DOM operations
      const fragment = document.createDocumentFragment();
      
      for (let i = 0; i < GRID_SIZE; i++) {
        for (let j = 0; j < GRID_SIZE; j++) {
          // Only create particles in a circular pattern (fewer particles)
          const distFromCenter = Math.sqrt(
            Math.pow(i - GRID_SIZE/2, 2) + 
            Math.pow(j - GRID_SIZE/2, 2)
          );
          
          // Skip some particles for better performance
          if (distFromCenter > GRID_SIZE/2 || Math.random() > 0.7) continue;
          
          const dot = document.createElement('div');
          const size = DOT_SIZE;
          dot.className = 'absolute rounded-full transition-all';
          dot.style.width = `${size}px`;
          dot.style.height = `${size}px`;
          dot.style.left = `${(j * 100) / GRID_SIZE}%`;
          dot.style.top = `${(i * 100) / GRID_SIZE}%`;
          
          // Simplified styling
          dot.style.backgroundColor = 'rgba(59, 130, 246, 0.1)';
          dot.style.transition = 'transform 0.7s ease-out, background-color 0.7s ease-out';
          
          fragment.appendChild(dot);
        }
      }
      
      gridRef.current.appendChild(fragment);
    }
    
    // Use setInterval instead of requestAnimationFrame for less frequent updates
    let lastUpdateTime = 0;
    let updateInterval;
    
    const updateDots = () => {
      const currentTime = Date.now();
      if (currentTime - lastUpdateTime < UPDATE_INTERVAL) return;
      lastUpdateTime = currentTime;
      
      const dots = gridRef.current.children;
      
      for (let i = 0; i < dots.length; i++) {
        const dot = dots[i];
        const dotRect = dot.getBoundingClientRect();
        const dotCenterX = dotRect.left + dotRect.width / 2;
        const dotCenterY = dotRect.top + dotRect.height / 2;
        
        // Calculate distance from mouse
        const dx = mousePosition.x - dotCenterX;
        const dy = mousePosition.y - dotCenterY;
        const distance = Math.sqrt(dx * dx + dy * dy);
        
        if (distance < MAX_DISTANCE) {
          // Simplified calculation with less math operations
          const scale = 1 + (1 - distance / MAX_DISTANCE) * 2.5;
          const force = (MAX_DISTANCE - distance) / MAX_DISTANCE;
          const translateX = -dx * force * 0.1;
          const translateY = -dy * force * 0.1;
          
          dot.style.transform = `translate(${translateX}px, ${translateY}px) scale(${scale})`;
          dot.style.backgroundColor = `rgba(59, 130, 246, ${0.1 + force * 0.3})`;
          dot.style.zIndex = "10";
        } else {
          // Static state when not interacting - no constant animation
          dot.style.transform = 'translate(0, 0) scale(1)';
          dot.style.backgroundColor = 'rgba(59, 130, 246, 0.1)';
          dot.style.zIndex = "1";
        }
      }
    };
    
    // Use mousemove event instead of continuous animation
    const handleMouseMove = () => {
      requestAnimationFrame(updateDots);
    };
    
    window.addEventListener('mousemove', handleMouseMove);
    
    // Initial update
    updateDots();
    
    // Clean up
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      clearInterval(updateInterval);
    };
  }, [mousePosition]);
  
  return (
    <div 
      ref={gridRef}
      className="absolute inset-0 bg-gradient-to-br from-blue-50/80 to-indigo-50/80 dark:from-gray-900 dark:to-gray-900"
    ></div>
  );
};

export default ParticleGridBackground;