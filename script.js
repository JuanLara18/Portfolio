document.addEventListener('DOMContentLoaded', () => {
    // ======= Elementos DOM =======
    const header = document.querySelector('header');
    const navLinks = document.querySelectorAll('nav a');
    const sections = document.querySelectorAll('section');
    const heroSection = document.querySelector('.hero');
    const experienceCards = document.querySelectorAll('.experience-card');
    const projectCards = document.querySelectorAll('.project-card');

    // ======= Animación de entrada para el hero =======
    const animateHero = () => {
        const heroContent = document.querySelector('.hero-content');
        heroContent.style.opacity = '0';
        heroContent.style.transform = 'translateY(20px)';
        
        setTimeout(() => {
            heroContent.style.transition = 'all 0.8s ease-out';
            heroContent.style.opacity = '1';
            heroContent.style.transform = 'translateY(0)';
        }, 200);
    };

    // ======= Efecto Parallax para el hero =======
    window.addEventListener('scroll', () => {
        const scrolled = window.pageYOffset;
        heroSection.style.transform = `translateY(${scrolled * 0.08}px)`;
    });

    // ======= Navegación Mejorada =======
    const smoothScroll = (target, duration = 800) => {
        const targetPosition = target.getBoundingClientRect().top + window.pageYOffset;
        const startPosition = window.pageYOffset;
        const distance = targetPosition - startPosition;
        let startTime = null;

        const animation = currentTime => {
            if (startTime === null) startTime = currentTime;
            const timeElapsed = currentTime - startTime;
            const run = ease(timeElapsed, startPosition, distance, duration);
            window.scrollTo(0, run);
            if (timeElapsed < duration) requestAnimationFrame(animation);
        };

        // Función de facilitación
        const ease = (t, b, c, d) => {
            t /= d / 2;
            if (t < 1) return c / 2 * t * t + b;
            t--;
            return -c / 2 * (t * (t - 2) - 1) + b;
        };

        requestAnimationFrame(animation);
    };

    // Navegación suave mejorada
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', e => {
            e.preventDefault();
            const targetId = anchor.getAttribute('href');
            const targetSection = document.querySelector(targetId);
            if (targetSection) {
                smoothScroll(targetSection);
                // Actualizar URL sin recargar
                history.pushState(null, null, targetId);
            }
        });
    });

    // ======= Observador de Intersección Mejorado =======
    const observerOptions = {
        root: null,
        threshold: 0.2,
        rootMargin: '50px 0px 50px 0px'
    };

    const sectionObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            // Una vez que el elemento es visible, lo mantenemos visible
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
                // Dejar de observar el elemento una vez que aparece
                sectionObserver.unobserve(entry.target);
            }
        });
    }, observerOptions);

    // ======= Animaciones de Cards =======
    const cardObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('card-visible');
            }
        });
    }, {
        threshold: 1.05
    });

    // Inicializar observadores y animaciones
    sections.forEach(section => {
        section.style.opacity = '0';
        section.style.transform = 'translateY(20px)';
        section.style.transition = 'all 0.6s ease-out';
        sectionObserver.observe(section);
    });

    experienceCards.forEach((card, index) => {
        card.style.opacity = '0';
        card.style.transform = 'translateX(-20px)';
        card.style.transition = `all 0.5s ease-out ${index * 0.2}s`;
        cardObserver.observe(card);
    });

    projectCards.forEach((card, index) => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(20px)';
        card.style.transition = `all 0.5s ease-out ${index * 0.2}s`;
        cardObserver.observe(card);
    });

    // ======= Efecto Header =======
    let lastScroll = 0;
    window.addEventListener('scroll', () => {
        const currentScroll = window.pageYOffset;
        
        // Efecto de aparición/desaparición del header
        if (currentScroll > lastScroll && currentScroll > 100) {
            header.style.transform = 'translateY(-100%)';
        } else {
            header.style.transform = 'translateY(0)';
        }
        
        // Efecto de blur y sombra
        if (currentScroll > 50) {
            header.style.backdropFilter = 'blur(10px)';
            header.style.boxShadow = '0 4px 6px rgba(0, 0, 0, 0.1)';
        } else {
            header.style.backdropFilter = 'blur(0px)';
            header.style.boxShadow = 'none';
        }

        lastScroll = currentScroll;
    });

    // ======= Loading Animation =======
    window.addEventListener('load', () => {
        document.body.classList.add('loaded');
        animateHero();
    });

    // ======= Hover Effects =======
    projectCards.forEach(card => {
        card.addEventListener('mouseenter', () => {
            card.style.transform = 'translateY(-10px) scale(1.02)';
            card.style.boxShadow = '0 20px 25px rgba(0, 0, 0, 0.15)';
        });

        card.addEventListener('mouseleave', () => {
            card.style.transform = 'translateY(0) scale(1)';
            card.style.boxShadow = 'var(--box-shadow)';
        });
    });

    // ======= Contact Form Analytics =======
    const trackContactClick = (type) => {
        console.log(`Contact interaction: ${type}`);
        // Aquí podrías integrar con Google Analytics u otra herramienta
    };

    document.querySelectorAll('.social-links a').forEach(link => {
        link.addEventListener('click', (e) => {
            const platform = link.getAttribute('href').includes('linkedin') ? 'LinkedIn' :
                           link.getAttribute('href').includes('github') ? 'GitHub' : 'Email';
            trackContactClick(platform);
        });
    });

    // Carousel Functionality
    const initializeCarousel = () => {
        const carousel = document.querySelector('.carousel-container');
        const cards = carousel.querySelectorAll('.training-card');
        const prevBtn = document.querySelector('.carousel-prev');
        const nextBtn = document.querySelector('.carousel-next');
        const indicatorsContainer = document.querySelector('.carousel-indicators');
        
        if (!carousel || !cards.length || !prevBtn || !nextBtn || !indicatorsContainer) {
            console.error('Required carousel elements not found');
            return;
        }

        let currentIndex = 0;
        const cardWidth = cards[0].offsetWidth + 24; // Width + gap
        const containerWidth = carousel.offsetWidth;
        const visibleCards = Math.floor(containerWidth / cardWidth);
        const maxIndex = Math.max(0, cards.length - visibleCards);

        // Clear existing indicators
        indicatorsContainer.innerHTML = '';

        // Create indicators
        for (let i = 0; i <= maxIndex; i++) {
            const indicator = document.createElement('div');
            indicator.classList.add('indicator');
            if (i === 0) indicator.classList.add('active');
            indicator.addEventListener('click', () => scrollToIndex(i));
            indicatorsContainer.appendChild(indicator);
        }

        const updateIndicators = () => {
            const indicators = indicatorsContainer.querySelectorAll('.indicator');
            indicators.forEach((indicator, index) => {
                indicator.classList.toggle('active', index === currentIndex);
            });
        };

        const updateButtons = () => {
            prevBtn.style.opacity = currentIndex === 0 ? '0.5' : '1';
            prevBtn.disabled = currentIndex === 0;
            nextBtn.style.opacity = currentIndex === maxIndex ? '0.5' : '1';
            nextBtn.disabled = currentIndex === maxIndex;
        };

        const scrollToIndex = (index) => {
            currentIndex = Math.max(0, Math.min(index, maxIndex));
            const scrollLeft = currentIndex * cardWidth;
            
            carousel.scrollTo({
                left: scrollLeft,
                behavior: 'smooth'
            });
            
            updateIndicators();
            updateButtons();
        };

        // Event Listeners
        prevBtn.addEventListener('click', () => {
            if (currentIndex > 0) {
                scrollToIndex(currentIndex - 1);
            }
        });

        nextBtn.addEventListener('click', () => {
            if (currentIndex < maxIndex) {
                scrollToIndex(currentIndex + 1);
            }
        });

        // Handle scroll events
        let scrollTimeout;
        carousel.addEventListener('scroll', () => {
            clearTimeout(scrollTimeout);
            scrollTimeout = setTimeout(() => {
                const newIndex = Math.round(carousel.scrollLeft / cardWidth);
                if (newIndex !== currentIndex) {
                    currentIndex = newIndex;
                    updateIndicators();
                    updateButtons();
                }
            }, 100);
        });

        // Touch events for mobile
        let touchStartX = 0;
        let touchEndX = 0;

        carousel.addEventListener('touchstart', (e) => {
            touchStartX = e.changedTouches[0].screenX;
        });

        carousel.addEventListener('touchend', (e) => {
            touchEndX = e.changedTouches[0].screenX;
            const diff = touchStartX - touchEndX;
            
            if (Math.abs(diff) > 50) { // Minimum swipe distance
                if (diff > 0 && currentIndex < maxIndex) {
                    scrollToIndex(currentIndex + 1);
                } else if (diff < 0 && currentIndex > 0) {
                    scrollToIndex(currentIndex - 1);
                }
            }
        });

        // Window resize handling
        let resizeTimeout;
        window.addEventListener('resize', () => {
            clearTimeout(resizeTimeout);
            resizeTimeout = setTimeout(() => {
                const newContainerWidth = carousel.offsetWidth;
                const newVisibleCards = Math.floor(newContainerWidth / cardWidth);
                const newMaxIndex = Math.max(0, cards.length - newVisibleCards);
                
                if (newMaxIndex !== maxIndex) {
                    currentIndex = Math.min(currentIndex, newMaxIndex);
                    scrollToIndex(currentIndex);
                }
            }, 200);
        });

        // Initial setup
        updateButtons();
    };

    // Initialize carousel
    initializeCarousel();

});