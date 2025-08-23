import { useLocation, Link } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Github, Linkedin, FileText } from 'lucide-react';

const MobileMenu = ({ isOpen, onClose }) => {
	const location = useLocation();
  
	const menuVariants = {
		closed: { x: '100%', transition: { type: 'spring', stiffness: 300, damping: 30 } },
		open: { x: 0, transition: { type: 'spring', stiffness: 300, damping: 30 } }
	};
	const overlayVariants = {
		closed: { opacity: 0, backdropFilter: 'blur(0px)', transition: { duration: 0.3, when: 'afterChildren' } },
		open: { opacity: 1, backdropFilter: 'blur(5px)', transition: { duration: 0.3, when: 'beforeChildren' } }
	};
	const getMobileLinkClass = (path) => {
		const isActive = path === '/' ? location.pathname === '/' : location.pathname.startsWith(path);
		return `flex items-center space-x-2 py-4 px-4 rounded-lg transition-colors touch-target ${
			isActive 
				? 'text-blue-600 dark:text-blue-400 font-medium bg-blue-50 dark:bg-blue-900/20' 
				: 'text-gray-800 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-800'
		}`;
	};

	return (
		<>
			<AnimatePresence>
				{isOpen && (
					<motion.div 
						initial="closed"
						animate="open"
						exit="closed"
						variants={overlayVariants}
						className="fixed inset-0 bg-gray-900/70 backdrop-blur-md z-40"
						onClick={onClose}
					/>
				)}
			</AnimatePresence>
      
			<motion.div 
				variants={menuVariants}
				initial="closed"
				animate={isOpen ? 'open' : 'closed'}
				className="fixed top-0 right-0 w-4/5 sm:w-3/4 max-w-sm h-full bg-white dark:bg-gray-900 z-50 transform md:hidden shadow-2xl border-l border-gray-200 dark:border-gray-700 flex flex-col"
			>
				<div className="p-4 sm:p-5 flex flex-col h-full">
					<div className="flex items-center justify-between mb-4 sm:mb-6 border-b border-gray-100 dark:border-gray-800 pb-4 sm:pb-5">
						<Link 
							to="/" 
							className="text-lg sm:text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-indigo-600 dark:from-blue-400 dark:to-indigo-400 touch-target"
							onClick={onClose}
						>
							Menu
						</Link>
						<button onClick={onClose} aria-label="Close menu" className="p-2 rounded-full hover:bg-gray-100 dark:hover:bg-gray-800 text-gray-700 dark:text-gray-300 transition-colors touch-target">
							<X size={20} />
						</button>
					</div>
          
					<nav className="flex flex-col space-y-1 mb-6 sm:mb-8">
						<Link to="/" className={getMobileLinkClass('/')} onClick={onClose}>
							<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="flex-shrink-0"><path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path><polyline points="9 22 9 12 15 12 15 22"></polyline></svg>
							<span>Home</span>
						</Link>
						<Link to="/about" className={getMobileLinkClass('/about')} onClick={onClose}>
							<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="flex-shrink-0"><circle cx="12" cy="8" r="5"></circle><path d="M20 21a8 8 0 1 0-16 0"></path></svg>
							<span>About</span>
						</Link>
						<Link to="/projects" className={getMobileLinkClass('/projects')} onClick={onClose}>
							<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="flex-shrink-0"><rect width="7" height="7" x="3" y="3" rx="1"></rect><rect width="7" height="7" x="14" y="3" rx="1"></rect><rect width="7" height="7" x="14" y="14" rx="1"></rect><rect width="7" height="7" x="3" y="14" rx="1"></rect></svg>
							<span>Projects</span>
						</Link>
						<Link to="/blog" className={getMobileLinkClass('/blog')} onClick={onClose}>
							<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="flex-shrink-0"><path d="M4 22h16a2 2 0 0 0 2-2V4a2 2 0 0 0-2-2H8a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2Z"></path><path d="M18 14h-8"></path><path d="M15 18h-5"></path><path d="M10 6h8v4h-8V6Z"></path></svg>
							<span>Blog</span>
						</Link>
					</nav>
          
					<div className="mt-auto">
						<div className="bg-gray-50 dark:bg-gray-800 rounded-xl p-4 sm:p-5">
							<h3 className="text-sm font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-4">Connect</h3>
							<div className="grid grid-cols-3 gap-2 sm:gap-3">
								<Link to="https://www.linkedin.com/in/julara/?locale=en_US" target="_blank" rel="noopener noreferrer" className="flex flex-col items-center justify-center bg-white dark:bg-gray-900 hover:bg-gray-100 dark:hover:bg-gray-800 text-gray-700 dark:text-gray-300 p-3 rounded-lg transition-colors touch-target">
									<Linkedin size={20} />
									<span className="text-xs mt-1">LinkedIn</span>
								</Link>
								<Link to="https://github.com/JuanLara18" target="_blank" rel="noopener noreferrer" className="flex flex-col items-center justify-center bg-white dark:bg-gray-900 hover:bg-gray-100 dark:hover:bg-gray-800 text-gray-700 dark:text-gray-300 p-3 rounded-lg transition-colors touch-target">
									<Github size={20} />
									<span className="text-xs mt-1">GitHub</span>
								</Link>
								<Link to="/documents/CV___EN.pdf" target="_blank" rel="noopener noreferrer" className="flex flex-col items-center justify-center bg-white dark:bg-gray-900 hover:bg-gray-100 dark:hover:bg-gray-800 text-gray-700 dark:text-gray-300 p-3 rounded-lg transition-colors touch-target">
									<FileText size={20} />
									<span className="text-xs mt-1">Resume</span>
								</Link>
							</div>
						</div>
						<div className="text-center text-gray-500 dark:text-gray-400 text-xs py-3 sm:py-4">© {new Date().getFullYear()} Juan Lara<div className="mt-1 text-xs">Computer Scientist & Mathematician</div></div>
					</div>
				</div>
			</motion.div>
		</>
	);
};

export default MobileMenu;
