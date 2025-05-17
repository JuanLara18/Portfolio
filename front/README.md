# Front-End Directory Structure

This directory contains the React front-end application for Juan Lara's portfolio website.

## Directory Structure

```
front/
├── public/                  # Static assets
│   ├── documents/           # PDF documents (Resume/CV)
│   ├── images/              # Images organized by category
│   │   ├── company-logos/   # Company logos used in experience section
│   │   ├── institutions/    # Educational institution logos
│   │   └── project-previews/# Project screenshots and previews
│   ├── favicon.ico          # Site favicon
│   ├── index.html           # HTML entry point
│   └── manifest.json        # Web app manifest
│
├── src/                     # Source code
│   ├── components/          # Reusable UI components
│   │   ├── Footer.jsx       # Site footer component
│   │   └── Navbar.jsx       # Navigation bar with dark mode toggle
│   │
│   ├── pages/               # Page components
│   │   ├── AboutPage.jsx    # About/bio page with education and experience
│   │   ├── LandingPage.jsx  # Homepage with hero section
│   │   └── ProjectsPage.jsx # Projects gallery with filtering
│   │
│   ├── App.js               # Main application component with routing
│   ├── App.css              # App-specific styles
│   ├── index.js             # JavaScript entry point
│   └── index.css            # Global styles with Tailwind directives
│
└── package.json             # Dependencies and scripts
```

## Key Files

- **App.js**: Contains the main application setup with React Router and dark mode state
- **components/Navbar.jsx**: Navigation with responsive mobile menu and theme switcher
- **pages/**: Contains the main page components that make up the site
- **index.css**: Imports Tailwind CSS utility classes

## Tech Stack

Built with React, React Router, Tailwind CSS, and Framer Motion for animations.

## Note

This is the front-end portion of the portfolio website. For installation instructions, project overview, and contribution guidelines, please refer to the main README in the repository root.