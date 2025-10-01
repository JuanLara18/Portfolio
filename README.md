# Juan Lara - Portfolio & Blog

A modern, responsive portfolio website showcasing my work as a Computer Scientist and Applied Mathematician, featuring an integrated blog system for sharing mathematical insights and research notes.

🌐 **Live Site**: [https://juanlara18.github.io/Portfolio](https://juanlara18.github.io/Portfolio)

## 🚀 Overview

This repository contains my personal portfolio website and blog, built with modern web technologies. The site serves as a professional showcase of my projects, experience, and academic background, while the integrated blog provides a platform for sharing mathematical curiosities and research insights.

### ✨ Key Features

- **Responsive Design**: Optimized for all devices and screen sizes
- **Dark/Light Mode**: System-aware theme switching with user preference persistence
- **Interactive Portfolio**: Filterable project showcase with detailed descriptions
- **Integrated Blog**: Markdown-based blog system with math support (KaTeX)
- **Professional Sections**: About, Experience, Education, Skills, and Awards
- **Smooth Animations**: Enhanced UX with Framer Motion animations
- **SEO Optimized**: Meta tags, structured data, and semantic HTML

## 🏗️ Current Architecture

```
Portfolio/
├── front/                          # React Frontend Application
│   ├── public/
│   │   ├── blog/                   # Blog System Assets
│   │   │   ├── posts/              # Markdown Blog Posts
│   │   │   │   ├── curiosities/    # Mathematical Curiosities
│   │   │   │   └── research/       # Research Notes
│   │   │   ├── figures/            # Post Images & Assets
│   │   │   ├── headers/            # Header Images
│   │   │   └── posts-manifest.json # Auto-generated Posts Index
│   │   ├── images/                 # Site Assets
│   │   └── documents/              # Static Documents (CV, etc.)
│   ├── src/
│   │   ├── components/             # Reusable UI Components
│   │   ├── pages/                  # Main Page Components
│   │   ├── utils/                  # Utility Functions
│   │   └── styles/                 # Styling Configuration
│   └── scripts/                    # Build & Automation Scripts
└── README.md                       # This File
```

## 🛠️ Technologies Used

### Frontend
- **React 18** - Component-based UI library
- **React Router 6** - Client-side routing
- **Tailwind CSS** - Utility-first CSS framework
- **Framer Motion** - Animation library
- **Lucide React** - Icon library

### Blog System
- **Markdown** - Content authoring format
- **KaTeX** - Math rendering ($\LaTeX$ support)
- **React Markdown** - Markdown rendering
- **Gray Matter** - Front matter parsing
- **Highlight.js** - Code syntax highlighting

### Development & Deployment
- **Create React App** - Build tooling
- **GitHub Actions** - CI/CD pipeline
- **GitHub Pages** - Static site hosting
- **ESLint** - Code linting
- **Date-fns** - Date manipulation

## 🚀 Getting Started

### Prerequisites
- Node.js (v16 or higher)
- npm or yarn
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/JuanLara18/Portfolio.git
   cd Portfolio
   ```

2. **Install dependencies**
   ```bash
   cd front
   npm install
   ```

3. **Generate blog posts manifest**
   ```bash
   npm run generate-manifest
   ```

4. **Start development server**
   ```bash
   npm start
   ```

5. **Open in browser**
   ```
   http://localhost:3000
   ```

### Development Workflow

#### Adding New Blog Posts

1. **Create markdown file**
   ```bash
   # For mathematical curiosities
   touch front/public/blog/posts/curiosities/your-post.md
   
   # For research notes
   touch front/public/blog/posts/research/your-post.md
   ```

2. **Add front matter and content**
   ```markdown
   ---
   title: "Your Post Title"
   date: "2025-01-25"
   excerpt: "Brief description"
   tags: ["Tag1", "Tag2"]
   headerImage: "/blog/headers/your-image.jpg"
   ---

   # Your Content Here
   
   Your markdown content with math support: $E = mc^2$
   ```

3. **Regenerate manifest**
   ```bash
   npm run generate-manifest
   ```

#### Available Scripts

- `npm start` - Start development server
- `npm run build` - Build for production
- `npm run generate-manifest` - Generate posts manifest
- `npm run generate-manifest:watch` - Watch for post changes
- `npm run deploy` - Deploy to GitHub Pages

## 📊 Future Roadmap

### Phase 1: Backend Development (Planned)
- **Analytics Dashboard**: Track visitor engagement and popular content
- **Contact Form Backend**: Handle form submissions and notifications
- **Content Management**: Admin interface for blog post management
- **Search API**: Enhanced search functionality with indexing

### Phase 2: Enhanced Features (Future)
- **Comments System**: Engage with readers on blog posts
- **Newsletter Integration**: Subscribe to updates and new posts
- **Performance Monitoring**: Real-time site performance analytics
- **Content Recommendations**: AI-powered related content suggestions

### Phase 3: Advanced Capabilities (Vision)
- **Multi-language Support**: Spanish and English content
- **Interactive Demos**: Embedded mathematical visualizations
- **API Documentation**: Expose data for third-party integrations
- **Mobile App**: Native mobile experience

## 🎯 Blog Categories

### Mathematical Curiosities
Explorations of games, puzzles, and mathematical phenomena including:
- Game theory applications
- Algorithmic complexity analysis
- Mathematical proofs and demonstrations
- Puzzle solving techniques

### Research Notes
Academic papers, studies, and research insights covering:
- Machine Learning and AI developments
- Natural Language Processing advances
- Computational mathematics applications
- Industry research analysis

## 🤝 Contributing

While this is a personal portfolio, I welcome suggestions and improvements:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/improvement`)
3. **Commit your changes** (`git commit -am 'Add improvement'`)
4. **Push to the branch** (`git push origin feature/improvement`)
5. **Open a Pull Request**

### Contribution Guidelines
- Follow existing code style and conventions
- Test changes thoroughly before submitting
- Update documentation for new features
- Ensure responsive design compatibility

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

**Juan Lara**
- 📧 Email: [larajuand@outlook.com](mailto:larajuand@outlook.com)
- 💼 LinkedIn: [linkedin.com/in/julara](https://www.linkedin.com/in/julara/)
- 🐱 GitHub: [github.com/JuanLara18](https://github.com/JuanLara18)
- 🌐 Website: [juanlara.dev](https://juanlara18.github.io/Portfolio)

## 🏆 Acknowledgments

- **Design Inspiration**: Modern portfolio best practices
- **Mathematical Content**: Personal research and academic work
- **Open Source Libraries**: React ecosystem and community tools
- **Deployment**: GitHub Pages for reliable hosting

---

### 📈 Project Stats

- **Frontend**: React 18 with modern JavaScript (ES2022)
- **Bundle Size**: Optimized for performance (~2MB total)
- **Lighthouse Score**: 95+ across all metrics
- **Browser Support**: All modern browsers (ES6+)
- **Mobile First**: Responsive design from 320px to 4K
- **Accessibility**: WCAG 2.1 AA compliant

### 🔧 Technical Highlights

- **Performance**: Code splitting and lazy loading
- **SEO**: Semantic HTML and meta optimization
- **Security**: CSP headers and secure practices
- **Maintainability**: Component-based architecture
- **Scalability**: Modular design for future expansion

Built with ❤️ and ☕ by Juan Lara