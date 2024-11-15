// Esperar a que el DOM esté completamente cargado
document.addEventListener('DOMContentLoaded', function() {
    // Inicialización de variables
    const header = document.querySelector('header');
    const navLinks = document.querySelectorAll('nav a');
    const sections = document.querySelectorAll('section');
    let lastScrollTop = 0;

    // ===============================
    // Navegación y Scroll
    // ===============================
    
    // Navegación suave para enlaces internos
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href');
            const targetSection = document.querySelector(targetId);
            
            if (targetSection) {
                targetSection.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // Control de header en scroll
    window.addEventListener('scroll', () => {
        const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
        
        // Añadir/remover clase scrolled al header
        if (scrollTop > 50) {
            header.classList.add('scrolled');
        } else {
            header.classList.remove('scrolled');
        }

        // Ocultar/mostrar header basado en dirección de scroll
        if (scrollTop > lastScrollTop && scrollTop > 100) {
            header.style.transform = 'translateY(-100%)';
        } else {
            header.style.transform = 'translateY(0)';
        }
        lastScrollTop = scrollTop;

        // Actualizar navegación activa basada en la sección visible
        updateActiveNavigation();
    });

    // ===============================
    // Animaciones de Entrada
    // ===============================
    
    // Observador para animaciones de entrada
    const fadeInElements = document.querySelectorAll('.fade-in');
    const observerOptions = {
        threshold: 0.2,
        rootMargin: '0px'
    };

    const appearOnScroll = new IntersectionObserver((entries, observer) => {
        entries.forEach(entry => {
            if (!entry.isIntersecting) return;
            entry.target.classList.add('visible');
            observer.unobserve(entry.target);
        });
    }, observerOptions);

    fadeInElements.forEach(element => {
        appearOnScroll.observe(element);
    });

    // ===============================
    // Funciones de Utilidad
    // ===============================
    
    // Actualizar navegación activa
    function updateActiveNavigation() {
        const scrollPosition = window.scrollY;

        sections.forEach(section => {
            const sectionTop = section.offsetTop - 100;
            const sectionBottom = sectionTop + section.offsetHeight;
            const sectionId = section.getAttribute('id');

            if (scrollPosition >= sectionTop && scrollPosition < sectionBottom) {
                navLinks.forEach(link => {
                    link.classList.remove('active');
                    if (link.getAttribute('href') === `#${sectionId}`) {
                        link.classList.add('active');
                    }
                });
            }
        });
    }

    // ===============================
    // Efectos de Typing
    // ===============================
    
    // Efecto de typing para el título principal
    const titleElement = document.querySelector('.hero h1');
    if (titleElement) {
        const text = titleElement.textContent;
        titleElement.textContent = '';
        let index = 0;

        function typeText() {
            if (index < text.length) {
                titleElement.textContent += text.charAt(index);
                index++;
                setTimeout(typeText, 100);
            }
        }

        // Iniciar efecto de typing después de un breve delay
        setTimeout(typeText, 500);
    }

    // ===============================
    // Manejo de Proyectos
    // ===============================
    
    // Añadir interactividad a las tarjetas de proyectos
    const projectCards = document.querySelectorAll('.project-card');
    projectCards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-10px)';
        });

        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0)';
        });
    });

    // ===============================
    // Inicialización de Contacto
    // ===============================
    
    // Manejar eventos de contacto
    const contactLinks = document.querySelectorAll('#contact a');
    contactLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            // Añadir analytics si es necesario
            console.log(`Contact click: ${this.getAttribute('href')}`);
        });
    });

    // ===============================
    // Optimización de Rendimiento
    // ===============================
    
    // Debounce para eventos de scroll
    function debounce(func, wait) {
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

    // Aplicar debounce al evento de scroll
    const debouncedScroll = debounce(() => {
        updateActiveNavigation();
    }, 100);

    window.addEventListener('scroll', debouncedScroll);

    // ===============================
    // Modo Oscuro (opcional)
    // ===============================
    
    // Detectar preferencia de modo oscuro del sistema
    const prefersDarkScheme = window.matchMedia('(prefers-color-scheme: dark)');
    
    function toggleDarkMode(e) {
        if (e.matches) {
            document.body.classList.add('dark-mode');
        } else {
            document.body.classList.remove('dark-mode');
        }
    }

    prefersDarkScheme.addListener(toggleDarkMode);
});