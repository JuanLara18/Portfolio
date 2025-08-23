import { useState, useEffect } from 'react';

export const useNavbarHeight = () => {
  const [navbarHeight, setNavbarHeight] = useState(80); // Default fallback

  useEffect(() => {
    const calculateNavbarHeight = () => {
      const navbar = document.querySelector('header[class*="fixed"]');
      if (navbar) {
        const height = navbar.getBoundingClientRect().height;
        setNavbarHeight(height);
      }
    };

    // Initial calculation
    calculateNavbarHeight();

    // Recalculate on resize with debounce
    let timeoutId;
    const handleResize = () => {
      clearTimeout(timeoutId);
      timeoutId = setTimeout(calculateNavbarHeight, 150);
    };

    window.addEventListener('resize', handleResize);
    
    // Also listen for orientation changes on mobile
    const handleOrientationChange = () => {
      setTimeout(calculateNavbarHeight, 300); // Delay for orientation change
    };
    window.addEventListener('orientationchange', handleOrientationChange);

    // Use ResizeObserver if available for more precise tracking
    let resizeObserver = null;
    if (window.ResizeObserver) {
      const navbar = document.querySelector('header[class*="fixed"]');
      if (navbar) {
        resizeObserver = new ResizeObserver(() => {
          calculateNavbarHeight();
        });
        resizeObserver.observe(navbar);
      }
    }

    // Cleanup function
    return () => {
      if (resizeObserver) {
        resizeObserver.disconnect();
      }
      window.removeEventListener('resize', handleResize);
      window.removeEventListener('orientationchange', handleOrientationChange);
      clearTimeout(timeoutId);
    };
  }, []);

  return navbarHeight;
};

export default useNavbarHeight;