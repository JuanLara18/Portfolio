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
        threshold: 1.1
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
            const cards = carousel?.querySelectorAll('.training-card');
            const prevBtn = document.querySelector('.carousel-prev');
            const nextBtn = document.querySelector('.carousel-next');
            const indicatorsContainer = document.querySelector('.carousel-indicators');

            if (!carousel || !cards?.length || !prevBtn || !nextBtn || !indicatorsContainer) {
                console.warn('Carousel elements not found');
                return;
            }

            let currentIndex = 0;
            const cardWidth = cards[0].offsetWidth + 24; // Width + gap
            const visibleCards = Math.floor(carousel.offsetWidth / cardWidth);
            const maxIndex = Math.max(0, cards.length - visibleCards);

            // Crear indicadores
            indicatorsContainer.innerHTML = '';
            for (let i = 0; i <= maxIndex; i++) {
                const indicator = document.createElement('div');
                indicator.className = `indicator${i === 0 ? ' active' : ''}`;
                indicator.addEventListener('click', () => scrollToIndex(i));
                indicatorsContainer.appendChild(indicator);
            }

            const updateIndicators = () => {
                indicatorsContainer.querySelectorAll('.indicator').forEach((ind, i) => {
                    ind.classList.toggle('active', i === currentIndex);
                });
            };

            const scrollToIndex = (index) => {
                currentIndex = Math.max(0, Math.min(index, maxIndex));
                carousel.scrollTo({
                    left: currentIndex * cardWidth,
                    behavior: 'smooth'
                });
                updateIndicators();
                
                // Actualizar estado de los botones
                prevBtn.disabled = currentIndex === 0;
                nextBtn.disabled = currentIndex === maxIndex;
                prevBtn.style.opacity = currentIndex === 0 ? '0.5' : '1';
                nextBtn.style.opacity = currentIndex === maxIndex ? '0.5' : '1';
            };

            // Event listeners para los botones
            prevBtn.addEventListener('click', () => scrollToIndex(currentIndex - 1));
            nextBtn.addEventListener('click', () => scrollToIndex(currentIndex + 1));

            // Inicializar estado de los botones
            scrollToIndex(0);
        };

        // Inicializar carrusel
        initializeCarousel();
    
    
        const hamburger = document.querySelector('.hamburger');
        // const navLinks = document.querySelector('.nav-links');
        const body = document.body;
        
        // Crear el overlay
        const overlay = document.createElement('div');
        overlay.className = 'nav-overlay';
        document.body.appendChild(overlay);
    
        // Toggle menú
        const toggleMenu = () => {
            hamburger.classList.toggle('active');
            navLinks.classList.toggle('active');
            overlay.classList.toggle('active');
            body.style.overflow = body.style.overflow === 'hidden' ? '' : 'hidden';
        };
    
        // Event listeners
        hamburger.addEventListener('click', toggleMenu);
        overlay.addEventListener('click', toggleMenu);
    
        // Cerrar menú al hacer click en un enlace
        const navItems = document.querySelectorAll('.nav-links a');
        navItems.forEach(item => {
            item.addEventListener('click', () => {
                if (navLinks.classList.contains('active')) {
                    toggleMenu();
                }
            });
        });
    
        // Cerrar menú al redimensionar la ventana
        window.addEventListener('resize', () => {
            if (window.innerWidth > 768 && navLinks.classList.contains('active')) {
                toggleMenu();
            }
        });
    
    
    });


    // Navegación
    const hamburger = document.querySelector('.hamburger');
    // const navLinks = document.querySelector('.nav-links');
    const overlay = document.createElement('div');
    
    if (hamburger && navLinks) {
        // Configurar overlay
        overlay.className = 'nav-overlay';
        document.body.appendChild(overlay);

        // Toggle menú
        const toggleMenu = () => {
            hamburger.classList.toggle('active');
            navLinks.classList.toggle('active');
            overlay.classList.toggle('active');
            document.body.style.overflow = hamburger.classList.contains('active') ? 'hidden' : '';
        };

        // Event listeners
        hamburger.addEventListener('click', toggleMenu);
        overlay.addEventListener('click', toggleMenu);

        // Cerrar menú al hacer click en enlaces
        navLinks.querySelectorAll('a').forEach(link => {
            link.addEventListener('click', () => {
                if (hamburger.classList.contains('active')) {
                    toggleMenu();
                }
            });
        });
    }


});