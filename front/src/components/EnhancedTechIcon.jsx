import { useState } from 'react';

const TechIcon = ({ icon: Icon, label, delay = 0 }) => {
  const [hovered, setHovered] = useState(false);
  
  return (
    <div 
      className={`flex flex-col items-center justify-center p-4 cursor-pointer opacity-100 transform transition-all duration-300 ease-in-out ${hovered ? 'scale-105' : ''}`}
      style={{ 
        // Handle the initial animation with inline styles
        // We can use CSS animation instead of framer-motion
        animation: `fadeIn 0.5s ease-out ${delay}s forwards`
      }}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      <div 
        className={`w-12 h-12 flex items-center justify-center rounded-lg mb-2 transition-all duration-300 ease-in-out
          bg-blue-50 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400 
          ${hovered ? 'transform -translate-y-1 shadow-lg' : 'shadow-md'}`}
      >
        <Icon size={24} className={`transition-transform duration-300 ${hovered ? 'scale-110' : ''}`} />
      </div>
      
      <span className="text-sm font-medium relative">
        <span 
          className={`transition-colors duration-300 ${hovered ? 'text-blue-500' : 'text-gray-700 dark:text-gray-300'}`}
        >
          {label}
        </span>
        
        {/* Animated underline */}
        <span 
          className={`absolute bottom-0 left-0 right-0 h-0.5 bg-blue-500 transform origin-left transition-transform duration-300 ease-out
            ${hovered ? 'scale-x-100' : 'scale-x-0'}`}
        />
      </span>
    </div>
  );
};

// Add the required CSS animation
const styles = `
@keyframes fadeIn {
  from {
    opacity: 0;
    transform: translateY(20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}
`;

const TechIconWithStyles = (props) => {
  return (
    <>
      <style jsx>{styles}</style>
      <TechIcon {...props} />
    </>
  );
};

export default TechIconWithStyles;