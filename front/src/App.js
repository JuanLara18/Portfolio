import { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { TransitionProvider, Navbar } from './components/layout';
import LandingPage from './pages/LandingPage';
import AboutPage from './pages/AboutPage';
import ProjectsPage from './pages/ProjectsPage';
import BlogHomePage from './pages/BlogHomePage';
import BlogPostPage from './pages/BlogPostPage';
import BlogCategoryPage from './pages/BlogCategoryPage';

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

  // Derive a safe basename from PUBLIC_URL (path only, no origin)
  const basename = (() => {
    const raw = process.env.PUBLIC_URL;
    if (!raw) return '/';
    try {
      // If it's an absolute URL, extract pathname; if it's a path, use as-is
      const url = raw.startsWith('http') ? new URL(raw) : new URL(raw, window.location.origin);
      const path = url.pathname || '/';
      // Ensure no trailing slash except root
      return path !== '/' && path.endsWith('/') ? path.slice(0, -1) : path || '/';
    } catch {
      return '/';
    }
  })();

  return (
    <Router 
      basename={basename}
      future={{ v7_startTransition: true, v7_relativeSplatPath: true }}
    >
      <TransitionProvider>
        <div className="App min-h-screen bg-white dark:bg-gray-900 text-gray-900 dark:text-gray-100">
          <Navbar 
            darkMode={darkMode} 
            toggleDarkMode={toggleDarkMode} 
          />
          
          {/* Main content with top padding for navbar */}
          <div className="pt-20">
            <Routes>
              <Route path="/" element={<LandingPage />} />
              <Route path="/about" element={<AboutPage />} />
              <Route path="/projects" element={<ProjectsPage />} />
              
              {/* Blog routes */}
              <Route path="/blog" element={<BlogHomePage />} />
              <Route path="/blog/category/:category" element={<BlogCategoryPage />} />
              <Route path="/blog/tag/:tag" element={<BlogCategoryPage />} />
              <Route path="/blog/:category/:slug" element={<BlogPostPage />} />
              
              {/* Fallback route redirects to home */}
              <Route path="*" element={<LandingPage />} />
            </Routes>
          </div>
          {/* Footer intentionally omitted for now */}
        </div>
      </TransitionProvider>
    </Router>
  );
}

export default App;