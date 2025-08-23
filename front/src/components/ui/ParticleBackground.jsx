import { useEffect, useRef } from 'react';

// Simplified particle animation constants
const GRID_SIZE = 24;
const DOT_SIZE = 5;
const MAX_DISTANCE = 150;
const UPDATE_INTERVAL = 25;

const ParticleBackground = ({ mousePosition }) => {
  const gridRef = useRef(null);
  
  useEffect(() => {
    if (!gridRef.current) return;
    
    // Create grid dots if not already created
    if (gridRef.current.children.length === 0) {
      const fragment = document.createDocumentFragment();
      
      for (let i = 0; i < GRID_SIZE; i++) {
        for (let j = 0; j < GRID_SIZE; j++) {
          const distFromCenter = Math.sqrt(
            Math.pow(i - GRID_SIZE/2, 2) + 
            Math.pow(j - GRID_SIZE/2, 2)
          );
          
          if (distFromCenter > GRID_SIZE/2 || Math.random() > 0.7) continue;
          
          const dot = document.createElement('div');
          const size = DOT_SIZE;
          dot.className = 'absolute rounded-full transition-all';
          dot.style.width = `${size}px`;
          dot.style.height = `${size}px`;
          dot.style.left = `${(j * 100) / GRID_SIZE}%`;
          dot.style.top = `${(i * 100) / GRID_SIZE}%`;
          dot.style.backgroundColor = 'rgba(59, 130, 246, 0.1)';
          dot.style.transition = 'transform 0.7s ease-out, background-color 0.7s ease-out';
          
          fragment.appendChild(dot);
        }
      }
      
      gridRef.current.appendChild(fragment);
    }
    
    let lastUpdateTime = 0;
    
    const updateDots = () => {
      if (!gridRef.current || !gridRef.current.children || gridRef.current.children.length === 0) {
        return;
      }
      
      const currentTime = Date.now();
      if (currentTime - lastUpdateTime < UPDATE_INTERVAL) return;
      lastUpdateTime = currentTime;
      
      const dots = gridRef.current.children;
      
      for (let i = 0; i < dots.length; i++) {
        const dot = dots[i];
        if (!dot) continue;
        
        try {
          const dotRect = dot.getBoundingClientRect();
          const dotCenterX = dotRect.left + dotRect.width / 2;
          const dotCenterY = dotRect.top + dotRect.height / 2;
          
          const dx = mousePosition.x - dotCenterX;
          const dy = mousePosition.y - dotCenterY;
          const distance = Math.sqrt(dx * dx + dy * dy);
          
          if (distance < MAX_DISTANCE) {
            const scale = 1 + (1 - distance / MAX_DISTANCE) * 2.5;
            const force = (MAX_DISTANCE - distance) / MAX_DISTANCE;
            const translateX = -dx * force * 0.1;
            const translateY = -dy * force * 0.1;
            
            dot.style.transform = `translate(${translateX}px, ${translateY}px) scale(${scale})`;
            dot.style.backgroundColor = `rgba(59, 130, 246, ${0.1 + force * 0.3})`;
            dot.style.zIndex = "10";
          } else {
            dot.style.transform = 'translate(0, 0) scale(1)';
            dot.style.backgroundColor = 'rgba(59, 130, 246, 0.1)';
            dot.style.zIndex = "1";
          }
        } catch (error) {
          // Silently continue if there's an error with this dot
          continue;
        }
      }
    };
    
    const handleMouseMove = () => {
      requestAnimationFrame(updateDots);
    };
    
    window.addEventListener('mousemove', handleMouseMove);
    updateDots(); // Initial update
    
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
    };
  }, [mousePosition]);
  
  return (
    <div 
      ref={gridRef}
      className="absolute inset-0 bg-gradient-to-br from-blue-50/80 to-indigo-50/80 dark:from-gray-900 dark:to-gray-900"
    />
  );
};

export default ParticleBackground;