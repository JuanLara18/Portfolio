# Personal Portfolio Website

This repository contains the source code for my personal portfolio website, a Flask-based web application that showcases my professional experience, educational background, and technical projects. The website features a responsive design, dark/light theme switching, and analytics tracking capabilities.

## Project Overview

The portfolio website serves as a comprehensive platform to present my professional profile, including:

- Professional experience and research work at Harvard Business School
- Educational background from Universidad Nacional de Colombia
- Technical skills and certifications
- Featured projects and their demonstrations
- Contact information and professional links

## Technical Architecture

The application is built using the following technologies:

- **Backend**: Flask (Python 3.11)
- **Frontend**: HTML5, CSS3, JavaScript
- **Analytics**: Custom SQLite-based analytics system
- **Deployment**: Heroku platform

## Project Structure

```
portfolio/
├── assets/
│   ├── documents/
│   ├── icons/
│   └── images/
├── src/
│   ├── index.html
│   ├── script.js
│   └── styles.css
├── Procfile
├── requirements.txt
├── runtime.txt
└── server.py
```

## Features

### Core Functionality

- Responsive design that adapts to all device sizes
- Dynamic theme switching between light and dark modes
- Interactive project showcase with live demos
- Professional experience timeline
- Educational background with detailed course information
- Integrated contact form and social media links

### Analytics Implementation

The website includes a custom analytics system that tracks:

- Page visits and user interactions
- Device types and geographical distribution
- Time spent on different sections
- Interaction patterns with various elements

## Setup and Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/portfolio.git
   cd portfolio
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the development server:
   ```bash
   python server.py
   ```

The application will be available at `http://localhost:8000`.

## Deployment

The application is configured for deployment on Heroku. Key deployment files include:

- `Procfile`: Specifies the command to run the application
- `runtime.txt`: Defines the Python version
- `requirements.txt`: Lists all Python dependencies

For deployment updates, push changes to the main branch:
```bash
git push origin main
```

## Environment Variables

The application uses the following environment variables:

- `FLASK_APP`: Set to "server.py"
- `FLASK_ENV`: Set to "production" for deployment
- `PORT`: Automatically set by Heroku

## Browser Compatibility

The website is tested and optimized for:

- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)

## Development Guidelines

When contributing to this project:

1. Maintain consistent code formatting using the established style
2. Add descriptive comments for complex functionality
3. Test responsive design across different device sizes
4. Ensure cross-browser compatibility
5. Update documentation for any new features

## Performance Considerations

The website implements several optimization strategies:

- Lazy loading of images
- Minified CSS and JavaScript
- Optimized asset delivery
- Efficient database queries for analytics

## Contact

For questions or collaboration opportunities, please reach out through:

- Email: [larajuand@outlook.com](mailto:larajuand@outlook.com)
- LinkedIn: [julara](https://www.linkedin.com/in/julara/)
- GitHub: [JuanLara18](https://github.com/JuanLara18)