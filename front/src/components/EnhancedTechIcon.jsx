import { useState } from 'react';
import { motion } from 'framer-motion';

const EnhancedTechIcon = ({ icon: Icon, label, delay = 0 }) => {
  // Create state for hover animation
  const [hovered, setHovered] = useState(false);
  
  return (
    <motion.div 
      className="flex flex-col items-center justify-center p-4 cursor-pointer"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ 
        delay,
        duration: 0.5,
        type: "spring", 
        stiffness: 100,
        damping: 15 
      }}
      whileHover={{ scale: 1.05 }}
      onHoverStart={() => setHovered(true)}
      onHoverEnd={() => setHovered(false)}
    >
      <motion.div 
        className="w-12 h-12 flex items-center justify-center rounded-lg bg-blue-50 dark:bg-blue-900/30 mb-2 text-blue-600 dark:text-blue-400 relative overflow-hidden"
        animate={{ 
          y: hovered ? -5 : 0,
          boxShadow: hovered 
            ? "0 15px 25px -5px rgba(59, 130, 246, 0.5)" 
            : "0 4px 6px -1px rgba(59, 130, 246, 0.1)",
        }}
        transition={{ 
          type: "spring", 
          stiffness: 300, 
          damping: 20 
        }}
      >
        {/* Icon container with 3D rotation effect */}
        <motion.div
          animate={{ 
            rotateY: hovered ? 180 : 0,
            transition: { duration: 0.6 }
          }}
          style={{ transformStyle: "preserve-3d" }}
        >
          {/* Front side */}
          <motion.div
            style={{ 
              backfaceVisibility: "hidden", 
              position: hovered ? "absolute" : "relative"
            }}
            animate={{ opacity: hovered ? 0 : 1 }}
          >
            <Icon size={24} />
          </motion.div>
          
          {/* Back side */}
          <motion.div
            style={{ 
              backfaceVisibility: "hidden", 
              position: "absolute",
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              transform: "rotateY(180deg)"
            }}
            animate={{ opacity: hovered ? 1 : 0 }}
          >
            <Icon size={24} />
          </motion.div>
        </motion.div>
        
        {/* Add gradient reflection effect */}
        <motion.div 
          className="absolute inset-0 opacity-0 bg-gradient-to-r from-transparent via-white to-transparent skew-x-20"
          animate={{ 
            opacity: hovered ? 0.3 : 0,
            x: hovered ? ["-100%", "200%"] : "0%",
            transition: { 
              opacity: { duration: 0.2 },
              x: { duration: 1, ease: "easeOut" }
            }
          }}
        />
      </motion.div>
      
      <motion.span 
        className="text-sm text-gray-700 dark:text-gray-300 font-medium relative"
        animate={{ 
          y: hovered ? 2 : 0,
          color: hovered ? "#3B82F6" : "",
        }}
      >
        {label}
        <motion.div 
          className="absolute bottom-0 left-0 right-0 h-0.5 bg-blue-500 origin-left"
          initial={{ scaleX: 0 }}
          animate={{ scaleX: hovered ? 1 : 0 }}
          transition={{ duration: 0.3 }}
        />
      </motion.span>
    </motion.div>
  );
};

export default EnhancedTechIcon;