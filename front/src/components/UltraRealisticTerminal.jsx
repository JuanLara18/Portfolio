import { useState, useEffect } from 'react';

const EnhancedTypingTerminal = ({ text = "Welcome to the enhanced terminal experience! This simulation now features more realistic typing patterns, better mistake handling, natural pauses, and improved visual effects for an authentic coding experience." }) => {
  const [displayText, setDisplayText] = useState('');
  const [currentIndex, setCurrentIndex] = useState(0);
  const [showCursor, setShowCursor] = useState(true);
  const [isComplete, setIsComplete] = useState(false);
  const [isTyping, setIsTyping] = useState(true);
  const [isPaused, setIsPaused] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [deleteCount, setDeleteCount] = useState(0);

  useEffect(() => {
    if (currentIndex < text.length && !isPaused && !isDeleting) {
      const currentChar = text[currentIndex];
      
      // More sophisticated mistake logic
      const shouldMakeMistake = () => {
        // Higher chance for complex words or after long sequences
        const complexity = currentIndex > 10 && Math.random() > 0.94;
        const afterPunctuation = currentIndex > 0 && '.!?'.includes(text[currentIndex - 1]);
        return complexity || (afterPunctuation && Math.random() > 0.85);
      };
      
      // Natural pause logic
      const shouldPauseNow = () => {
        const afterPunctuation = currentIndex > 0 && '.!?'.includes(text[currentIndex - 1]);
        const beforeComplexChar = '{}[]()<>'.includes(currentChar);
        const randomThinking = Math.random() > 0.988;
        const afterComma = currentIndex > 0 && text[currentIndex - 1] === ',';
        
        return afterPunctuation || beforeComplexChar || randomThinking || (afterComma && Math.random() > 0.7);
      };
      
      // Handle mistakes
      if (shouldMakeMistake() && currentIndex > 3) {
        setIsDeleting(true);
        setDeleteCount(2 + Math.floor(Math.random() * 3)); // Delete 2-4 characters
        return;
      }
      
      // Handle natural pauses
      if (shouldPauseNow()) {
        setIsPaused(true);
        const pauseDuration = text[currentIndex - 1] === '.' ? 400 + Math.random() * 600 : 200 + Math.random() * 400;
        setTimeout(() => setIsPaused(false), pauseDuration);
        return;
      }
      
      // Calculate realistic typing speed
      const getTypingSpeed = () => {
        let baseSpeed = 45;
        
        // Character-specific adjustments
        if (currentChar === ' ') baseSpeed -= 15; // Faster spacebar
        if (currentChar.match(/[A-Z]/)) baseSpeed += 25; // Slower capitals
        if ('{}[]()<>'.includes(currentChar)) baseSpeed += 40; // Slower special chars
        if ('etaoinshrdlu'.includes(currentChar.toLowerCase())) baseSpeed -= 10; // Faster common letters
        if (currentIndex > 0 && text[currentIndex - 1] === ' ') baseSpeed += 15; // Slower after space
        
        // Add natural variation
        const variation = (Math.random() - 0.5) * 30;
        return Math.max(25, baseSpeed + variation);
      };
      
      const timeout = setTimeout(() => {
        setDisplayText(prev => prev + currentChar);
        setCurrentIndex(prev => prev + 1);
      }, getTypingSpeed());
      
      return () => clearTimeout(timeout);
    } 
    
    // Handle deletion for mistakes
    else if (isDeleting && deleteCount > 0) {
      const timeout = setTimeout(() => {
        setDisplayText(prev => prev.slice(0, -1));
        setCurrentIndex(prev => prev - 1);
        setDeleteCount(prev => prev - 1);
        
        if (deleteCount === 1) {
          setIsDeleting(false);
          // Small pause before continuing
          setTimeout(() => {}, 100);
        }
      }, 60 + Math.random() * 40);
      
      return () => clearTimeout(timeout);
    }
    
    // Completion handling
    else if (currentIndex >= text.length) {
      setIsTyping(false);
      setIsComplete(true);
    }
  }, [currentIndex, text, isPaused, isDeleting, deleteCount]);

  // Enhanced cursor blinking with different states
  useEffect(() => {
    const getBlinkSpeed = () => {
      if (isPaused) return 300; // Faster blink when paused (thinking)
      if (isDeleting) return 150; // Very fast blink when deleting
      if (isComplete) return 530; // Normal blink when done
      return 400; // Typing blink
    };
    
    const interval = setInterval(() => {
      setShowCursor(prev => !prev);
    }, getBlinkSpeed());
    
    return () => clearInterval(interval);
  }, [isPaused, isDeleting, isComplete]);

  return (
    <div 
      className="relative font-mono p-4 bg-gray-900 text-green-400 rounded-md overflow-hidden shadow-xl border border-gray-800 transition-all duration-700"
      style={{
        opacity: 1,
        transform: 'translateY(0px)',
        animation: 'terminalGlow 2s ease-in-out infinite alternate'
      }}
    >
      {/* Enhanced terminal header */}
      <div className="flex items-center gap-2 mb-2 pb-2 border-b border-gray-700">
        <div className="w-3 h-3 rounded-full bg-red-500 hover:bg-red-400 transition-all duration-200 cursor-pointer shadow-lg" 
             style={{boxShadow: '0 0 8px rgba(239, 68, 68, 0.4)'}} />
        <div className="w-3 h-3 rounded-full bg-yellow-500 hover:bg-yellow-400 transition-all duration-200 cursor-pointer shadow-lg"
             style={{boxShadow: '0 0 8px rgba(251, 191, 36, 0.4)'}} />
        <div className="w-3 h-3 rounded-full bg-green-500 hover:bg-green-400 transition-all duration-200 cursor-pointer shadow-lg"
             style={{boxShadow: '0 0 8px rgba(34, 197, 94, 0.4)'}} />
        <span className="ml-2 text-gray-400 text-sm font-medium">juan@lara ~ terminal</span>
        
        {/* Status indicator */}
        <div className="ml-auto flex items-center gap-2 text-xs">
          {isPaused && <span className="text-yellow-400 animate-pulse">● thinking...</span>}
          {isDeleting && <span className="text-red-400 animate-pulse">● correcting...</span>}
          {isTyping && !isPaused && !isDeleting && <span className="text-green-400">● typing...</span>}
          {isComplete && <span className="text-blue-400">● complete</span>}
        </div>
      </div>
      
      <div className="flex h-full">
        <span 
          className="text-blue-400 mr-2 font-bold transition-all duration-300"
          style={{
            opacity: 1,
            textShadow: '0 0 10px rgba(59, 130, 246, 0.5)'
          }}
        >$</span>
        
        <div className="min-h-[150px] flex-1">
          <div className="leading-relaxed">
            {displayText}
            
            {/* Enhanced cursor with different states */}
            {showCursor && (
              <span 
                className={`inline-block w-2 h-5 ml-1 transition-all duration-150 ${
                  isPaused ? 'bg-yellow-400 animate-pulse' :
                  isDeleting ? 'bg-red-400' :
                  'bg-green-400'
                }`}
                style={{
                  boxShadow: isPaused ? '0 0 8px rgba(251, 191, 36, 0.8)' :
                            isDeleting ? '0 0 8px rgba(239, 68, 68, 0.8)' :
                            '0 0 8px rgba(34, 197, 94, 0.8)',
                  animation: isPaused ? 'cursorPulse 1s ease-in-out infinite' : 'none'
                }}
              />
            )}
          </div>
          
          {/* Enhanced completion effect */}
          {isComplete && (
            <div 
              className="w-full h-8 bg-gradient-to-b from-transparent via-green-900/20 to-green-900/5 mt-4 rounded transition-all duration-1000"
              style={{
                opacity: 0.6,
                animation: 'completionGlow 2s ease-in-out'
              }}
            />
          )}
        </div>
      </div>
      
      {/* Enhanced scan lines with movement */}
      <div 
        className="absolute inset-0 bg-scan-lines opacity-[0.04] pointer-events-none" 
        style={{
          backgroundImage: `
            linear-gradient(0deg, transparent 0%, transparent 48%, rgba(32, 255, 32, 0.3) 49%, rgba(32, 255, 32, 0.3) 51%, transparent 52%, transparent 100%)
          `,
          backgroundSize: '100% 4px',
          animation: 'scanLines 0.1s linear infinite'
        }}
      />
      
      {/* Enhanced multiple glow layers */}
      <div className="absolute inset-0 bg-green-500/3 blur-md pointer-events-none" />
      <div className="absolute inset-0 bg-green-400/2 blur-lg pointer-events-none" />
      
      {/* Subtle screen flicker */}
      <div 
        className="absolute inset-0 bg-green-500/1 pointer-events-none"
        style={{
          animation: 'screenFlicker 3s ease-in-out infinite'
        }}
      />
      
      {/* Corner glow effects */}
      <div className="absolute top-0 left-0 w-32 h-32 bg-green-500/5 blur-xl rounded-full pointer-events-none" />
      <div className="absolute bottom-0 right-0 w-24 h-24 bg-blue-500/5 blur-xl rounded-full pointer-events-none" />
      
      <style jsx>{`
        @keyframes terminalGlow {
          0% { box-shadow: 0 0 20px rgba(34, 197, 94, 0.1); }
          100% { box-shadow: 0 0 30px rgba(34, 197, 94, 0.2), 0 0 40px rgba(34, 197, 94, 0.1); }
        }
        
        @keyframes scanLines {
          0% { transform: translateY(-100%); }
          100% { transform: translateY(100vh); }
        }
        
        @keyframes cursorPulse {
          0%, 100% { opacity: 1; transform: scale(1); }
          50% { opacity: 0.7; transform: scale(1.1); }
        }
        
        @keyframes completionGlow {
          0% { opacity: 0; transform: scaleY(0); }
          50% { opacity: 0.8; transform: scaleY(1.2); }
          100% { opacity: 0.3; transform: scaleY(1); }
        }
        
        @keyframes screenFlicker {
          0%, 98%, 100% { opacity: 1; }
          99% { opacity: 0.98; }
        }
      `}</style>
    </div>
  );
};

export default EnhancedTypingTerminal;