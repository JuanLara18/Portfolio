// Main portfolio functionality
const Portfolio = {
    // DOM Elements cache
    elements: {},

    // Initialize portfolio
    init() {
        this.cacheElements();
        this.bindEvents();
        this.initializeFeatures();
    },

    // Cache all DOM elements
    cacheElements() {
        this.elements = {
            header: document.querySelector('header'),
            hero: document.querySelector('.hero'),
            heroContent: document.querySelector('.hero-content'),
            nav: document.querySelector('nav'),
            navLinks: document.querySelector('.nav-links'),
            hamburger: document.querySelector('.hamburger'),
            sections: document.querySelectorAll('section'),
            experienceCards: document.querySelectorAll('.experience-card'),
            projectCards: document.querySelectorAll('.project-card'),
            carousel: {
                container: document.querySelector('.carousel-container'),
                cards: document.querySelectorAll('.training-card'),
                prevBtn: document.querySelector('.carousel-prev'),
                nextBtn: document.querySelector('.carousel-next'),
                indicators: document.querySelector('.carousel-indicators')
            },
            socialLinks: document.querySelectorAll('.social-links a')
        };
    },

    // Bind all event listeners
    bindEvents() {
        // Scroll events
        window.addEventListener('scroll', this.handleScroll.bind(this));
        window.addEventListener('load', this.handleLoad.bind(this));
        window.addEventListener('resize', this.debounce(this.handleResize.bind(this), 250));

        // Navigation
        this.initializeNavigation();

        // Cards hover effects
        this.initializeCardEffects();

        // Social links tracking
        this.initializeSocialTracking();
    },

    // Initialize all features
    initializeFeatures() {
        this.initializeHero();
        this.initializeObservers();
        this.initializeCarousel();
        this.initializeMobileMenu();
    },

    // Hero animations and effects
    initializeHero() {
        if (!this.elements.heroContent) return;

        // Entry animation
        this.elements.heroContent.style.opacity = '0';
        this.elements.heroContent.style.transform = 'translateY(20px)';

        setTimeout(() => {
            this.elements.heroContent.style.transition = 'all 0.8s ease-out';
            this.elements.heroContent.style.opacity = '1';
            this.elements.heroContent.style.transform = 'translateY(0)';
        }, 200);

        // Parallax effect
        this.parallaxEffect = this.debounce((scrolled) => {
            if (this.elements.hero) {
                this.elements.hero.style.transform = `translateY(${scrolled * 0.08}px)`;
            }
        }, 10);
    },

    // Intersection Observer setup
    initializeObservers() {
        const observerOptions = {
            root: null,
            threshold: 0.2,
            rootMargin: '50px 0px'
        };

        // Sections observer
        const sectionObserver = new IntersectionObserver(entries => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('visible');
                    entry.target.style.opacity = '1';
                    entry.target.style.transform = 'translateY(0)';
                    sectionObserver.unobserve(entry.target);
                }
            });
        }, observerOptions);

        // Initialize sections and cards
        this.elements.sections.forEach(section => {
            section.style.opacity = '0';
            section.style.transform = 'translateY(20px)';
            section.style.transition = 'all 0.6s ease-out';
            sectionObserver.observe(section);
        });

        // Card animations
        [this.elements.experienceCards, this.elements.projectCards].forEach((cards, isProject) => {
            cards.forEach((card, index) => {
                card.style.opacity = '0';
                card.style.transform = isProject ? 'translateY(20px)' : 'translateX(-20px)';
                card.style.transition = `all 0.5s ease-out ${index * 0.2}s`;
                sectionObserver.observe(card);
            });
        });
    },

    // Carousel functionality
    initializeCarousel() {
        const { container, cards, prevBtn, nextBtn, indicators } = this.elements.carousel;
        if (!container || !cards.length) return;

        let currentIndex = 0;
        const cardWidth = cards[0].offsetWidth + 24;
        const visibleCards = Math.floor(container.offsetWidth / cardWidth);
        const maxIndex = Math.max(0, cards.length - visibleCards);

        // Create indicators
        indicators.innerHTML = Array.from({ length: maxIndex + 1 }, (_, i) => 
            `<div class="indicator${i === 0 ? ' active' : ''}" data-index="${i}"></div>`
        ).join('');

        // Handle navigation
        const scrollToIndex = (index) => {
            currentIndex = Math.max(0, Math.min(index, maxIndex));
            container.scrollTo({ left: currentIndex * cardWidth, behavior: 'smooth' });
            
            // Update UI
            indicators.querySelectorAll('.indicator').forEach((ind, i) => 
                ind.classList.toggle('active', i === currentIndex));
            prevBtn.style.opacity = currentIndex === 0 ? '0.5' : '1';
            nextBtn.style.opacity = currentIndex === maxIndex ? '0.5' : '1';
        };

        // Event listeners
        [prevBtn, nextBtn, indicators].forEach(el => {
            if (!el) return;
            el.addEventListener('click', (e) => {
                if (e.target.classList.contains('indicator')) {
                    scrollToIndex(parseInt(e.target.dataset.index));
                } else if (e.target === prevBtn) {
                    scrollToIndex(currentIndex - 1);
                } else if (e.target === nextBtn) {
                    scrollToIndex(currentIndex + 1);
                }
            });
        });
    },

    // Mobile menu
    initializeMobileMenu() {
        const { hamburger, navLinks } = this.elements;
        if (!hamburger || !navLinks) return;

        const overlay = document.createElement('div');
        overlay.className = 'nav-overlay';
        document.body.appendChild(overlay);

        const toggleMenu = () => {
            hamburger.classList.toggle('active');
            navLinks.classList.toggle('active');
            overlay.classList.toggle('active');
            document.body.style.overflow = navLinks.classList.contains('active') ? 'hidden' : '';
        };

        [hamburger, overlay].forEach(el => 
            el.addEventListener('click', toggleMenu));

        navLinks.querySelectorAll('a').forEach(link => 
            link.addEventListener('click', () => {
                if (navLinks.classList.contains('active')) toggleMenu();
            }));
    },

    // Smooth scroll navigation
    initializeNavigation() {
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', (e) => {
                e.preventDefault();
                const target = document.querySelector(anchor.getAttribute('href'));
                if (target) {
                    const headerOffset = this.elements.header?.offsetHeight || 0;
                    const elementPosition = target.offsetTop;
                    window.scrollTo({
                        top: elementPosition - headerOffset,
                        behavior: 'smooth'
                    });
                }
            });
        });
    },

    // Card hover effects
    initializeCardEffects() {
        this.elements.projectCards.forEach(card => {
            card.addEventListener('mouseenter', () => {
                card.style.transform = 'translateY(-10px) scale(1.02)';
                card.style.boxShadow = '0 20px 25px rgba(0, 0, 0, 0.15)';
            });

            card.addEventListener('mouseleave', () => {
                card.style.transform = 'translateY(0) scale(1)';
                card.style.boxShadow = 'var(--box-shadow)';
            });
        });
    },

    // Social link tracking
    initializeSocialTracking() {
        this.elements.socialLinks.forEach(link => {
            link.addEventListener('click', () => {
                const platform = link.href.includes('linkedin') ? 'LinkedIn' :
                               link.href.includes('github') ? 'GitHub' : 'Email';
                console.log(`Contact interaction: ${platform}`);
                // Add your analytics implementation here
            });
        });
    },

    // Scroll handler
    handleScroll() {
        const scrolled = window.pageYOffset;
        
        // Parallax effect
        this.parallaxEffect(scrolled);

        // Header effects
        if (this.elements.header) {
            this.elements.header.style.transform = scrolled > 100 ? 
                'translateY(-100%)' : 'translateY(0)';
            this.elements.header.style.backdropFilter = scrolled > 50 ? 
                'blur(10px)' : 'blur(0px)';
            this.elements.header.style.boxShadow = scrolled > 50 ? 
                '0 4px 6px rgba(0, 0, 0, 0.1)' : 'none';
        }
    },

    // Load handler
    handleLoad() {
        document.body.classList.add('loaded');
        this.initializeHero();
    },

    // Resize handler
    handleResize() {
        this.initializeCarousel();
    },

    // Utility: Debounce function
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }
};

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => Portfolio.init());