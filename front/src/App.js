import { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Footer from './components/Footer';
import LandingPage from './pages/LandingPage';
import AboutPage from './pages/AboutPage';
import ProjectsPage from './pages/ProjectsPage';

// Import other pages when they're created
// import ContactPage from './pages/ContactPage';

function App() {
  // Dark mode state
  const [darkMode, setDarkMode] = useState(false);
  
  // Initialize darkMode based on user preference or localStorage
  useEffect(() => {
    // Check if user has a preference in localStorage
    const savedTheme = localStorage.getItem('theme');
    
    if (savedTheme) {
      setDarkMode(savedTheme === 'dark');
    } else {
      // Check if user prefers dark mode
      const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
      setDarkMode(prefersDark);
    }
  }, []);
  
  // Toggle dark mode and save preference
  const toggleDarkMode = () => {
    const newDarkMode = !darkMode;
    setDarkMode(newDarkMode);
    localStorage.setItem('theme', newDarkMode ? 'dark' : 'light');
  };
  
  // Apply dark mode class to document
  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [darkMode]);

  return (
    <Router>
      <div className="App min-h-screen bg-white dark:bg-gray-900 text-gray-900 dark:text-gray-100">
        <Navbar darkMode={darkMode} toggleDarkMode={toggleDarkMode} />
        
        {/* Main content with top padding for navbar */}
        <div className="pt-20">
          <Routes>
            <Route path="/" element={<LandingPage />} />
            <Route path="/about" element={<AboutPage />} />
            <Route path="/projects" element={<ProjectsPage />} />
            {/* Add routes for other pages as they are created */}
            {/* <Route path="/contact" element={<ContactPage />} /> */}
            
            {/* Fallback route redirects to home */}
            <Route path="*" element={<LandingPage />} />
          </Routes>
        </div>
      </div>
    </Router>
  );
}

export default App;