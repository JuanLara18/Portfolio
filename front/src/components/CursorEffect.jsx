import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';

const CursorEffect = () => {
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
  const [cursorPoints, setCursorPoints] = useState([]);
  const maxPoints = 5; // Number of trailing points
  
  // Track mouse movement
  useEffect(() => {
    const handleMouseMove = (e) => {
      setMousePosition({ x: e.clientX, y: e.clientY });
    };
    
    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, []);
  
  // Update cursor trail
  useEffect(() => {
    // Add new point to the beginning
    const updatedPoints = [mousePosition, ...cursorPoints.slice(0, maxPoints - 1)];
    setCursorPoints(updatedPoints);
  }, [mousePosition, cursorPoints]);
  
  // Only show on desktop
  if (typeof window !== 'undefined' && window.innerWidth < 768) return null;
  
  return (
    <>
      {/* Main cursor */}
      <motion.div 
        className="fixed w-8 h-8 rounded-full pointer-events-none z-50 mix-blend-difference"
        style={{ 
          x: mousePosition.x - 16, 
          y: mousePosition.y - 16,
          backgroundColor: 'white'
        }}
        animate={{ 
          scale: [1, 1.2, 1],
          transition: { duration: 1, repeat: Infinity }
        }}
      />
      
      {/* Cursor trail */}
      {cursorPoints.map((point, index) => (
        point.x > 0 && (
          <motion.div 
            key={index}
            className="fixed w-3 h-3 rounded-full pointer-events-none z-50 mix-blend-difference"
            style={{ 
              x: point.x - 6, 
              y: point.y - 6,
              backgroundColor: 'white',
              opacity: 1 - (index / maxPoints)
            }}
            transition={{ duration: 0.1 }}
          />
        )
      ))}
    </>
  );
};

export default CursorEffect;