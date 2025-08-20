import { memo } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import remarkGfm from 'remark-gfm';
import remarkFrontmatter from 'remark-frontmatter';
import rehypeKatex from 'rehype-katex';
import rehypeHighlight from 'rehype-highlight';
import 'katex/dist/katex.min.css';
import 'highlight.js/styles/github-dark.css';

const BlogMarkdownRenderer = memo(({ content, className = "", baseImagePath = "" }) => {
	// Safely extract plain text from React children trees
	const extractText = (node) => {
		if (node == null) return '';
		if (typeof node === 'string' || typeof node === 'number') return String(node);
		if (Array.isArray(node)) return node.map(extractText).join('');
		if (typeof node === 'object' && 'props' in node) return extractText(node.props?.children);
		return '';
	};

	const withPublicUrl = (p) => {
		if (!p) return '';
		const base = process.env.PUBLIC_URL || '';
		if (p.startsWith('http')) return p;
		if (p.startsWith('/')) return `${base}${p}`;
		return `${base}/${p}`;
	};
	// Custom components for markdown elements
	const components = {
		// Enhanced headings with IDs for linking
	h1: ({ children, ...props }) => (
			<h1 
		id={slugify(extractText(children))}
				className="text-3xl md:text-4xl font-bold mb-6 mt-8 text-gray-900 dark:text-gray-100 border-b border-gray-200 dark:border-gray-700 pb-2"
				{...props}
			>
				{children}
			</h1>
		),
	h2: ({ children, ...props }) => (
			<h2 
		id={slugify(extractText(children))}
				className="text-2xl md:text-3xl font-bold mb-4 mt-8 text-gray-900 dark:text-gray-100"
				{...props}
			>
				{children}
			</h2>
		),
	h3: ({ children, ...props }) => (
			<h3 
		id={slugify(extractText(children))}
				className="text-xl md:text-2xl font-semibold mb-3 mt-6 text-gray-900 dark:text-gray-100"
				{...props}
			>
				{children}
			</h3>
		),
	h4: ({ children, ...props }) => (
			<h4 
		id={slugify(extractText(children))}
				className="text-lg md:text-xl font-semibold mb-3 mt-5 text-gray-900 dark:text-gray-100"
				{...props}
			>
				{children}
			</h4>
		),
	h5: ({ children, ...props }) => (
			<h5 
		id={slugify(extractText(children))}
				className="text-base md:text-lg font-semibold mb-2 mt-4 text-gray-900 dark:text-gray-100"
				{...props}
			>
				{children}
			</h5>
		),
	h6: ({ children, ...props }) => (
			<h6 
		id={slugify(extractText(children))}
				className="text-sm md:text-base font-semibold mb-2 mt-4 text-gray-700 dark:text-gray-300"
				{...props}
			>
				{children}
			</h6>
		),
    
		// Enhanced paragraphs
		p: ({ children, ...props }) => (
			<p className="mb-4 leading-relaxed text-gray-700 dark:text-gray-300 text-base md:text-lg" {...props}>
				{children}
			</p>
		),
    
		// Enhanced images with responsive design and error handling
		img: ({ src = '', alt, title, ...props }) => {
			let imageSrc = '';
			if (src.startsWith('http')) {
				imageSrc = src;
			} else if (src.startsWith('/')) {
				imageSrc = withPublicUrl(src);
			} else if (src.startsWith('figures/')) {
				// Allow markdown to reference `figures/...` from blog root
				imageSrc = withPublicUrl(`/blog/${src}`);
			} else {
				// Default: relative to provided baseImagePath
				const base = baseImagePath?.endsWith('/') ? baseImagePath.slice(0, -1) : baseImagePath;
				imageSrc = withPublicUrl(`${base}/${src}`.replace(/\/+/, '/'));
			}
      
			return (
				<div className="my-8">
					<img
						src={imageSrc}
						alt={alt || ''}
						title={title}
						className="w-full h-auto rounded-lg shadow-lg border border-gray-200 dark:border-gray-700"
						loading="lazy"
						onError={(e) => {
							// Fallback to a default placeholder image instead of disappearing
							const fallback = withPublicUrl('/blog/headers/default.jpg');
							if (e.target.src !== fallback) {
								e.target.src = fallback;
							} else {
								e.target.style.display = 'none';
							}
						}}
						{...props}
					/>
					{(alt || title) && (
						<div className="mt-2 text-sm text-gray-600 dark:text-gray-400 text-center italic">
							{title || alt}
						</div>
					)}
				</div>
			);
		},
    
		// Enhanced links with external link detection
		a: ({ href, children, ...props }) => {
			const isExternal = href && (href.startsWith('http') || href.startsWith('mailto:'));
			const linkClass = "text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300 underline decoration-2 underline-offset-2 transition-colors";
      
			if (isExternal) {
				return (
					<a 
						href={href} 
						className={linkClass}
						target="_blank" 
						rel="noopener noreferrer"
						{...props}
					>
						{children}
						<svg 
							className="inline ml-1 w-3 h-3" 
							fill="none" 
							viewBox="0 0 24 24" 
							stroke="currentColor"
						>
							<path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
						</svg>
					</a>
				);
			}
      
			return <a href={href} className={linkClass} {...props}>{children}</a>;
		},
    
		// Enhanced code blocks
		code: ({ inline, className, children, ...props }) => {
			if (inline) {
				return (
					<code 
						className="px-1.5 py-0.5 bg-gray-100 dark:bg-gray-800 text-gray-800 dark:text-gray-200 rounded text-sm font-mono border"
						{...props}
					>
						{children}
					</code>
				);
			}
      
			return (
				<div className="my-6">
					<code 
						className={`${className} block p-4 bg-gray-50 dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-700 overflow-x-auto text-sm leading-relaxed`}
						{...props}
					>
						{children}
					</code>
				</div>
			);
		},
    
		// Enhanced pre blocks (for code syntax highlighting)
		pre: ({ children, ...props }) => (
			<pre 
				className="my-6 p-4 bg-gray-50 dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-700 overflow-x-auto"
				{...props}
			>
				{children}
			</pre>
		),
    
		// Enhanced lists
		ul: ({ children, ...props }) => (
			<ul className="mb-4 ml-6 space-y-2 list-disc text-gray-700 dark:text-gray-300" {...props}>
				{children}
			</ul>
		),
		ol: ({ children, ...props }) => (
			<ol className="mb-4 ml-6 space-y-2 list-decimal text-gray-700 dark:text-gray-300" {...props}>
				{children}
			</ol>
		),
		li: ({ children, ...props }) => (
			<li className="leading-relaxed" {...props}>
				{children}
			</li>
		),
    
		// Enhanced blockquotes
		blockquote: ({ children, ...props }) => (
			<blockquote 
				className="my-6 pl-6 border-l-4 border-blue-500 bg-blue-50 dark:bg-blue-900/20 py-4 pr-4 rounded-r-lg"
				{...props}
			>
				<div className="text-gray-700 dark:text-gray-300 italic">
					{children}
				</div>
			</blockquote>
		),
    
		// Enhanced tables
		table: ({ children, ...props }) => (
			<div className="my-6 overflow-x-auto">
				<table className="w-full border-collapse border border-gray-300 dark:border-gray-600 rounded-lg overflow-hidden" {...props}>
					{children}
				</table>
			</div>
		),
		thead: ({ children, ...props }) => (
			<thead className="bg-gray-100 dark:bg-gray-800" {...props}>
				{children}
			</thead>
		),
		th: ({ children, ...props }) => (
			<th className="px-4 py-3 text-left font-semibold text-gray-900 dark:text-gray-100 border border-gray-300 dark:border-gray-600" {...props}>
				{children}
			</th>
		),
		td: ({ children, ...props }) => (
			<td className="px-4 py-3 text-gray-700 dark:text-gray-300 border border-gray-300 dark:border-gray-600" {...props}>
				{children}
			</td>
		),
    
		// Horizontal rule
		hr: ({ ...props }) => (
			<hr className="my-8 border-gray-300 dark:border-gray-600" {...props} />
		),
	};

	return (
		<div className={`prose prose-lg max-w-none dark:prose-invert ${className}`}>
			<ReactMarkdown
				components={components}
				remarkPlugins={[remarkMath, remarkGfm, remarkFrontmatter]}
				rehypePlugins={[rehypeKatex, rehypeHighlight]}
			>
				{content}
			</ReactMarkdown>
		</div>
	);
});

// Helper function for creating URL-friendly slugs
function slugify(text) {
	return text
		.toString()
		.normalize('NFD')
		.replace(/[\u0300-\u036f]/g, '')
		.toLowerCase()
		.trim()
		.replace(/\s+/g, '-')
		.replace(/[^\w\-]+/g, '')
		.replace(/\-\-+/g, '-');
}

export default BlogMarkdownRenderer;
