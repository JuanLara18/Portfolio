import { Link } from 'react-router-dom';
import { Github, Linkedin, Mail, ExternalLink } from 'lucide-react';

const Footer = () => {
  const currentYear = new Date().getFullYear();
  
  return (
    <footer className="py-12 bg-slate-900 text-white">
      <div className="container mx-auto px-6">
        <div className="max-w-5xl mx-auto">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-10">
            {/* Column 1: Logo & Description */}
            <div className="md:col-span-2">
              <Link to="/" className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-primary-500 to-secondary-500">
                Juan Lara
              </Link>
              <p className="mt-4 text-slate-400 max-w-md">
                Computer Scientist & Applied Mathematician specializing in Machine Learning, 
                AI Agents, and Natural Language Processing.
              </p>
              <div className="flex space-x-4 mt-6">
                <a 
                  href="https://github.com/JuanLara18" 
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-slate-400 hover:text-white transition-colors"
                  aria-label="GitHub Profile"
                >
                  <Github size={20} />
                </a>
                <a 
                  href="https://www.linkedin.com/in/julara/?locale=en_US" 
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-slate-400 hover:text-white transition-colors"
                  aria-label="LinkedIn Profile"
                >
                  <Linkedin size={20} />
                </a>
                <a 
                  href="mailto:larajuand@outlook.com"
                  className="text-slate-400 hover:text-white transition-colors"
                  aria-label="Email"
                >
                  <Mail size={20} />
                </a>
              </div>
            </div>
            
            {/* Column 2: Quick Links */}
            <div>
              <h3 className="text-white font-semibold mb-4 text-sm uppercase tracking-wider">Navigation</h3>
              <ul className="space-y-2">
                <li>
                  <Link to="/" className="text-slate-400 hover:text-white transition-colors">Home</Link>
                </li>
                <li>
                  <Link to="/about" className="text-slate-400 hover:text-white transition-colors">About</Link>
                </li>
                <li>
                  <Link to="/projects" className="text-slate-400 hover:text-white transition-colors">Projects</Link>
                </li>
                <li>
                  <a 
                    href="https://blog.juanlara.dev" 
                    className="text-slate-400 hover:text-white transition-colors inline-flex items-center gap-1"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    Blog <ExternalLink size={14} />
                  </a>
                </li>
              </ul>
            </div>
            
            {/* Column 3: Resources */}
            <div>
              <h3 className="text-white font-semibold mb-4 text-sm uppercase tracking-wider">Resources</h3>
              <ul className="space-y-2">
                <li>
                  <a 
                    href="/assets/documents/CV___EN.pdf" 
                    className="text-slate-400 hover:text-white transition-colors"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    Resume / CV
                  </a>
                </li>
                <li>
                  <a 
                    href="https://github.com/JuanLara18?tab=repositories" 
                    className="text-slate-400 hover:text-white transition-colors"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    GitHub Repositories
                  </a>
                </li>
                <li>
                  <a 
                    href="mailto:larajuand@outlook.com?subject=Contact%20from%20Portfolio" 
                    className="text-slate-400 hover:text-white transition-colors"
                  >
                    Contact Me
                  </a>
                </li>
              </ul>
            </div>
          </div>
          
          <div className="border-t border-slate-800 mt-12 pt-8 flex flex-col md:flex-row justify-between items-center">
            <p className="text-slate-500 text-sm mb-4 md:mb-0">
              &copy; {currentYear} Juan Lara. All rights reserved.
            </p>
            <p className="text-slate-500 text-sm">
              Built with React & Tailwind CSS
            </p>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;