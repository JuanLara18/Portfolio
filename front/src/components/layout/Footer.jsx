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
              
              {/* Social Links */}
              <div className="flex space-x-5 mt-6">
                <a 
                  href="https://github.com/juanlara18" 
                  className="p-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white transition-all duration-200"
                  target="_blank"
                  rel="noopener noreferrer"
                  aria-label="GitHub Profile"
                >
                  <Github size={20} />
                </a>
                <a 
                  href="https://linkedin.com/in/juan-camilo-lara-cruz" 
                  className="p-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white transition-all duration-200"
                  target="_blank"
                  rel="noopener noreferrer"
                  aria-label="LinkedIn Profile"
                >
                  <Linkedin size={20} />
                </a>
                <a 
                  href="mailto:juancamilolara18@gmail.com" 
                  className="p-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white transition-all duration-200"
                  aria-label="Email Contact"
                >
                  <Mail size={20} />
                </a>
              </div>
            </div>

            {/* Column 2: Quick Links */}
            <div>
              <h4 className="font-semibold mb-4 text-slate-200">Navigation</h4>
              <ul className="space-y-2">
                <li>
                  <Link to="/" className="text-slate-400 hover:text-white transition-colors">
                    Home
                  </Link>
                </li>
                <li>
                  <Link to="/about" className="text-slate-400 hover:text-white transition-colors">
                    About
                  </Link>
                </li>
                <li>
                  <Link to="/projects" className="text-slate-400 hover:text-white transition-colors">
                    Projects
                  </Link>
                </li>
                <li>
                  <Link to="/blog" className="text-slate-400 hover:text-white transition-colors">
                    Blog
                  </Link>
                </li>
              </ul>
            </div>

            {/* Column 3: Blog Categories */}
            <div>
              <h4 className="font-semibold mb-4 text-slate-200">Blog</h4>
              <ul className="space-y-2">
                <li>
                  <Link to="/blog/category/research" className="text-slate-400 hover:text-white transition-colors">
                    Research
                  </Link>
                </li>
                <li>
                  <Link to="/blog/category/curiosities" className="text-slate-400 hover:text-white transition-colors">
                    Curiosities
                  </Link>
                </li>
                <li>
                  <a 
                    href="mailto:juancamilolara18@gmail.com" 
                    className="text-slate-400 hover:text-white transition-colors inline-flex items-center"
                  >
                    Contact
                    <ExternalLink size={14} className="ml-1" />
                  </a>
                </li>
              </ul>
            </div>
          </div>

          {/* Bottom Bar */}
          <div className="border-t border-slate-800 mt-12 pt-8 flex flex-col md:flex-row justify-between items-center text-slate-400">
            <p>
              © {currentYear} Juan Lara. All rights reserved.
            </p>
            <p className="text-sm mt-2 md:mt-0">
              Built with React & Tailwind CSS
            </p>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;