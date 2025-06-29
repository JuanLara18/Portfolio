import { useState, useEffect, useRef } from 'react';
import { motion, useScroll, useTransform, useInView } from 'framer-motion';
import { Link } from 'react-router-dom';
import { 
  ExternalLink, 
  GraduationCap, 
  Briefcase, 
  Award, 
  Code, 
  Database, 
  BookOpen, 
  Server, 
  BrainCircuit, 
  LineChart, 
  Globe, 
  Mail, 
  Phone,
  MapPin,
  Github,
  BarChart,
  Terminal,
  Cloud,
  Layers,
  Box
} from 'lucide-react';

// Animation variants
const fadeInUp = {
  hidden: { opacity: 0, y: 30 },
  visible: { 
    opacity: 1, 
    y: 0,
    transition: { duration: 0.7, ease: [0.22, 1, 0.36, 1] }
  }
};

const fadeInRight = {
  hidden: { opacity: 0, x: -30 },
  visible: { 
    opacity: 1, 
    x: 0,
    transition: { duration: 0.7, ease: [0.22, 1, 0.36, 1] }
  }
};

const fadeInLeft = {
  hidden: { opacity: 0, x: 30 },
  visible: { 
    opacity: 1, 
    x: 0,
    transition: { duration: 0.7, ease: [0.22, 1, 0.36, 1] }
  }
};

const staggerContainer = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.15,
      delayChildren: 0.2
    }
  }
};

// Skill component with animated progress bar
const SkillBar = ({ name, level, icon: Icon, color = "blue" }) => {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: "-50px" });
  
  // Convert level (0-5) to percentage
  const percentage = (level / 5) * 100;
  
  // Color mappings for different UI elements
  const colorClasses = {
    blue: {
      iconBg: "bg-blue-100 dark:bg-blue-900/30",
      iconText: "text-blue-600 dark:text-blue-400",
      progressBar: "bg-blue-600 dark:bg-blue-500"
    },
    indigo: {
      iconBg: "bg-indigo-100 dark:bg-indigo-900/30",
      iconText: "text-indigo-600 dark:text-indigo-400",
      progressBar: "bg-indigo-600 dark:bg-indigo-500"
    },
    green: {
      iconBg: "bg-green-100 dark:bg-green-900/30",
      iconText: "text-green-600 dark:text-green-400",
      progressBar: "bg-green-600 dark:bg-green-500"
    },
    red: {
      iconBg: "bg-red-100 dark:bg-red-900/30",
      iconText: "text-red-600 dark:text-red-400",
      progressBar: "bg-red-600 dark:bg-red-500"
    },
    yellow: {
      iconBg: "bg-yellow-100 dark:bg-yellow-900/30",
      iconText: "text-yellow-600 dark:text-yellow-400",
      progressBar: "bg-yellow-600 dark:bg-yellow-500"
    },
    teal: {
      iconBg: "bg-teal-100 dark:bg-teal-900/30",
      iconText: "text-teal-600 dark:text-teal-400",
      progressBar: "bg-teal-600 dark:bg-teal-500"
    },
    orange: {
      iconBg: "bg-orange-100 dark:bg-orange-900/30", 
      iconText: "text-orange-600 dark:text-orange-400",
      progressBar: "bg-orange-600 dark:bg-orange-500"
    },
    purple: {
      iconBg: "bg-purple-100 dark:bg-purple-900/30",
      iconText: "text-purple-600 dark:text-purple-400",
      progressBar: "bg-purple-600 dark:bg-purple-500"
    }
  };
  
  // Get color classes or fallback to blue if color is not in our mapping
  const classes = colorClasses[color] || colorClasses.blue;
  
  return (
    <div className="mb-6" ref={ref}>
      <div className="flex items-center mb-2">
        <div className={`w-8 h-8 rounded-md ${classes.iconBg} flex items-center justify-center mr-3`}>
          <Icon size={18} className={classes.iconText} />
        </div>
        <span className="text-gray-800 dark:text-gray-200 font-medium">{name}</span>
      </div>
      <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
        <motion.div 
          className={`h-full ${classes.progressBar} rounded-full`}
          initial={{ width: 0 }}
          animate={{ width: isInView ? `${percentage}%` : 0 }}
          transition={{ duration: 1, ease: "easeOut", delay: 0.2 }}
        />
      </div>
    </div>
  );
};

// Experience card component
const ExperienceCard = ({ 
  role, 
  company, 
  period, 
  location, 
  description, 
  responsibilities, 
  skills, 
  logo 
}) => {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: "-100px" });
  
  return (
    <motion.div 
      ref={ref}
      initial={{ opacity: 0, y: 30 }}
      animate={isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 30 }}
      transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
      className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 mb-10 border border-gray-100 dark:border-gray-700 relative overflow-hidden group"
    >
      <div className="absolute top-0 right-0 w-40 h-40 bg-blue-50/50 dark:bg-blue-900/10 rounded-full -mr-20 -mt-20 z-0 transform group-hover:scale-110 transition-transform duration-500"></div>
      
      <div className="relative z-10">
        <div className="flex flex-col md:flex-row md:items-center mb-4 gap-4">
          <div className="w-24 h-24 md:w-24 md:h-24 rounded-lg overflow-hidden flex-shrink-0 bg-white p-2 shadow-md">
            <img 
              src={`${process.env.PUBLIC_URL}/images/company-logos/${logo}`} 
              alt={company} 
              className="w-full h-full object-contain"
            />
          </div>
          
          <div>
            <h3 className="text-xl font-bold text-gray-900 dark:text-white">{role}</h3>
            <div className="text-lg text-blue-600 dark:text-blue-400 font-medium">{company}</div>
            <div className="flex flex-wrap items-center text-sm text-gray-600 dark:text-gray-400 mt-1 gap-2">
              <span>{period}</span>
              <span className="w-1 h-1 rounded-full bg-gray-400"></span>
              <span>{location}</span>
            </div>
          </div>
        </div>
        
        <p className="text-gray-700 dark:text-gray-300 mb-4 leading-relaxed">{description}</p>
        
        {responsibilities && (
          <div className="mb-4">
            <h4 className="text-sm uppercase tracking-wider text-gray-600 dark:text-gray-400 font-semibold mb-2">Key Responsibilities</h4>
            <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
              {responsibilities.map((item, index) => (
                <li key={index} className="leading-relaxed">{item}</li>
              ))}
            </ul>
          </div>
        )}
        
        {skills && (
          <div className="flex flex-wrap gap-2 mt-4">
            {skills.map((skill, index) => (
              <span 
                key={index} 
                className="px-3 py-1 bg-blue-50 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 rounded-md text-sm"
              >
                {skill}
              </span>
            ))}
          </div>
        )}
      </div>
    </motion.div>
  );
};

// Education card component
const EducationCard = ({ 
  degree, 
  institution, 
  period, 
  location, 
  focus, 
  gpa, 
  certificateLink, 
  logo 
}) => {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: "-100px" });
  
  return (
    <motion.div 
      ref={ref}
      initial={{ opacity: 0, y: 30 }}
      animate={isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 30 }}
      transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
      className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 mb-8 border border-gray-100 dark:border-gray-700 relative overflow-hidden"
    >
      <div className="absolute bottom-0 left-0 w-40 h-40 bg-blue-50/50 dark:bg-blue-900/10 rounded-full -ml-20 -mb-20 z-0"></div>
      
      <div className="relative z-10">
        <div className="flex flex-col md:flex-row md:items-center mb-6 gap-4">
          <div className="w-24 h-24 md:w-24 md:h-24 rounded-lg overflow-hidden flex-shrink-0 bg-white p-2 shadow-md">
            <img 
              src={`${process.env.PUBLIC_URL}/images/institutions/${logo}`} 
              alt={institution} 
              className="w-full h-full object-contain"
            />
          </div>
          
          <div>
            <h3 className="text-xl font-bold text-gray-900 dark:text-white">{degree}</h3>
            <div className="text-lg text-blue-600 dark:text-blue-400">{institution}</div>
            <div className="flex flex-wrap items-center text-sm text-gray-600 dark:text-gray-400 mt-1 gap-2">
              <span>{period}</span>
              <span className="w-1 h-1 rounded-full bg-gray-400"></span>
              <span>{location}</span>
            </div>
          </div>
        </div>
        
        <div className="flex flex-wrap gap-3 mb-4">
          {focus && (
            <div className="px-4 py-2 bg-purple-50 dark:bg-purple-900/20 text-purple-700 dark:text-purple-300 rounded-md text-sm flex items-center">
              <BookOpen size={14} className="mr-1.5" />
              <span>Focus: {focus}</span>
            </div>
          )}
          
          {gpa && (
            <div className="px-4 py-2 bg-green-50 dark:bg-green-900/20 text-green-700 dark:text-green-300 rounded-md text-sm flex items-center">
              <Award size={14} className="mr-1.5" />
              <span>GPA: {gpa}</span>
            </div>
          )}
        </div>
        
        {certificateLink && (
          <a 
            href={certificateLink}
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center text-blue-600 dark:text-blue-400 hover:underline gap-1 group"
          >
            <span>View Certificate</span>
            <ExternalLink size={14} className="transform group-hover:translate-x-1 transition-transform duration-300" />
          </a>
        )}
      </div>
    </motion.div>
  );
};

// Course/Training card component
const CourseCard = ({ 
  title, 
  provider, 
  date, 
  duration,
  description, 
  certificateLink, 
  certificateId,
  topics,
  logo 
}) => {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: "-100px" });
  
  return (
    <motion.div 
      ref={ref}
      initial={{ opacity: 0, y: 30 }}
      animate={isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 30 }}
      transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
      className="bg-white dark:bg-gray-800 rounded-xl shadow-md p-5 border border-gray-100 dark:border-gray-700 h-full relative overflow-hidden group"
    >
      <div className="absolute top-0 right-0 w-24 h-24 bg-indigo-50/50 dark:bg-indigo-900/10 rounded-full -mr-12 -mt-12 z-0 transform group-hover:scale-110 transition-transform duration-500"></div>
      
      <div className="relative z-10">
        <div className="flex items-start gap-3 mb-3">
          <div className="w-16 h-16 rounded-lg overflow-hidden flex-shrink-0 bg-white p-1 shadow-sm">
            <img 
              src={`${process.env.PUBLIC_URL}/images/institutions/${logo}`} 
              alt={provider} 
              className="w-full h-full object-contain"
            />
          </div>
          
          <div>
            <h3 className="text-base font-bold text-gray-900 dark:text-white line-clamp-2">{title}</h3>
            <div className="text-sm text-gray-600 dark:text-gray-400">{provider}</div>
          </div>
        </div>
        
        <div className="flex flex-wrap gap-2 mb-3 text-xs">
          <span className="px-2 py-1 bg-blue-50 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 rounded-md">
            {date}
          </span>
          
          {duration && (
            <span className="px-2 py-1 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-md">
              {duration}
            </span>
          )}
        </div>
        
        {description && (
          <p className="text-sm text-gray-700 dark:text-gray-300 mb-3 line-clamp-2">{description}</p>
        )}
        
        {topics && topics.length > 0 && (
          <div className="flex flex-wrap gap-1.5 mb-3">
            {topics.map((topic, index) => (
              <span 
                key={index}
                className="px-2 py-0.5 bg-indigo-50 dark:bg-indigo-900/20 text-indigo-700 dark:text-indigo-300 rounded text-xs"
              >
                {topic}
              </span>
            ))}
          </div>
        )}
        
        {certificateLink && (
          <div className="mt-auto pt-2">
            <a 
              href={certificateLink}
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center text-blue-600 dark:text-blue-400 text-sm hover:underline gap-1 group"
            >
              <span>Verify Certificate {certificateId && `(ID: ${certificateId})`}</span>
              <ExternalLink size={12} className="transform group-hover:translate-x-1 transition-transform duration-300" />
            </a>
          </div>
        )}
      </div>
    </motion.div>
  );
};

export default function AboutPage() {
  const { scrollY } = useScroll();
  const heroRef = useRef(null);
  const isHeroInView = useInView(heroRef);
  
  // Transform values based on scroll position
  const heroOpacity = useTransform(scrollY, [260, 500], [1, 0.98]);
  const heroScale = useTransform(scrollY, [260, 500], [1, 0.98]);
  
// Experience data
const experiences = [
  {
    role: "Research Assistant",
    company: "Harvard Business School",
    period: "Sep 2022 - Present",
    location: "Boston, USA (Remote)",
    description:
      "Integrated machine learning into research workflows and co-designed rigorous mathematical frameworks for organizational economics, in collaboration with the Digital Reskilling Lab.",
    responsibilities: [
      "Integrated clustering techniques, XGBoost models, and NLP pipelines to empirically validate theoretical frameworks using extensive company datasets.",
      "Collaborated with the Digital Reskilling Lab to bridge quantitative research and practical business applications, generating actionable insights for reskilling strategies and organizational decision-making.",
      "Co-designed economic models of firm behavior—covering organizational hierarchies, exclusive contract dynamics, and technology shocks—and produced proofs and simulations for an upcoming working paper."
    ],
    // Skills: ["Python", "Machine Learning", "NLP", "XGBoost"],
    logo: "hbs-logo.png"
  },
  {
    role: "Data Scientist",
    company: "Ipsos",
    period: "Feb 2024 - Jan 2025",
    location: "Bogota, D.C., Colombia (Hybrid)",
    description:
      "Developed multi-platform ML applications and led the creation of TextInsight, earning recognition as Total Ops Star Employee for LATAM.",
    responsibilities: [
      "Developed multi-platform applications embedding machine learning models for segmentation tasks, leveraging geospatial data and deploying scalable solutions on Google Cloud Platform to enhance field operations efficiency.",
      "Streamlined analytical workflows and reporting with user-friendly tools, reducing manual effort, improving operational decision-making, and securing the Total Ops Star Employee award for LATAM.",
      "Led the design and implementation of TextInsight, a Python library for automated multilingual text analysis using LLMs and advanced NetworkX visualizations, enabling faster, more accurate survey insights across Latin America."
    ],
    logo: "ipsos-logo.jpg"
  }
];

  
  // Education data
  const education = [
    {
      degree: "B.S. in Computer Science",
      institution: "Universidad Nacional de Colombia",
      period: "Feb 2019 - Nov 2023",
      location: "Bogotá D.C.",
      focus: "Machine Learning",
      gpa: "4.7/5.0",
      certificateLink: "https://drive.google.com/file/d/1bp6QKeEqpOeCBIBKsst0IwQpr48nmjoi/view?usp=sharing",
      logo: "unal-logo.png"
    },
    {
      degree: "B.S. in Mathematics",
      institution: "Universidad Nacional de Colombia",
      period: "Feb 2018 - Jun 2022",
      location: "Bogotá D.C.",
      focus: "Applied Mathematics",
      gpa: "4.7/5.0",
      certificateLink: "https://drive.google.com/file/d/1RW4Q3Kca8rfMUJpejdwlmtTTiA5YgYwU/view?usp=sharing",
      logo: "unal-logo.png"
    },
    // {
    //   degree: "Technical Baccalaureate in Business Administration",
    //   institution: "Centro Educativo los Andes",
    //   period: "Feb 2015 - Nov 2017",
    //   location: "Bogotá D.C.",
    //   focus: "Entrepreneurship and Investigation",
    //   gpa: "4.5",
    //   logo: "school-logo.png" // You'll need to add an appropriate logo or use a placeholder
    // },
    // {
    //   degree: "Technician in Maintenance of Computer Equipment",
    //   institution: "Servicio Nacional de Aprendizaje - SENA",
    //   period: "Nov 2015 - Dec 2016",
    //   location: "Bogotá D.C.",
    //   focus: "Corrective Software",
    //   gpa: "4.6/5.0",
    //   certificateLink: "https://drive.google.com/file/d/1beMFGTbBiNhCdUMJnjLYQfCetz7V3xCW/view",
    //   logo: "sena-logo.png" // You'll need to add an appropriate logo or use a placeholder
    // }
  ];
  
  // Courses/Training data
  const courses = [
    {
      title: "AI Agents Fundamentals",
      provider: "Hugging Face",
      date: "February 2025",
      description: "Training on foundational concepts and practical implementation of AI agents using Hugging Face tools and frameworks.",
      certificateLink: "https://huggingface.co/datasets/agents-course/certificates/resolve/main/certificates/juanlara/2025-02-19.png",
      certificateId: "juanlara",
      topics: ["AI Agents", "Multi-agent Systems", "Transformers"],
      logo: "hugging_face-logo.png"
    },
    {
      title: "Artificial Intelligence Expert Certificate (CAIEC)",
      provider: "Certiprof",
      date: "November 2024",
      description: "Advanced-level certification focusing on artificial intelligence concepts, methodologies, and best practices.",
      certificateLink: "https://www.credly.com/badges/TLZVDQTVTGG-XWHHHQPTQ-RDJFLDLRK",
      certificateId: "TLZVDQTVTGG-XWHHHQPTQ-RDJFLDLRK",
      topics: ["AI", "Neural Networks", "Machine Learning"],
      logo: "certiprof_CAIEC-logo.png"
    },
    {
      title: "Artificial Intelligence Bootcamp",
      provider: "Talento Tech Cymetria",
      date: "May-October 2024",
      duration: "159 hours",
      description: "Intensive training in AI and machine learning, covering cutting-edge algorithms and deep learning model construction.",
      certificateLink: "https://certificados.talentotech.co/?cert=2518458921#pdf",
      certificateId: "2518458921",
      topics: ["Deep Learning", "Neural Networks", "PyTorch"],
      logo: "cymetria-logo.png"
    },
    {
      title: "DevOps Certification",
      provider: "Platzi",
      date: "October 2024",
      description: "Program covering Docker, Swarm, GitHub Actions, GitLab, Jenkins, Azure DevOps, and MLOps practices for continuous integration and deployment.",
      certificateLink: "https://platzi.com/p/larajuan/learning-path/8353-cloud-devops/diploma/detalle/",
      certificateId: "cc4cfe8a-d78a-4883-8a75-ca90931151f6",
      topics: ["Docker", "GitHub Actions", "MLOps"],
      logo: "platzi-logo.png"
    },
    {
      title: "Algorithmic Toolbox",
      provider: "Coursera",
      date: "2023",
      description: "Course focused on algorithm design and implementation techniques for solving computational problems efficiently.",
      certificateLink: "https://www.coursera.org/account/accomplishments/certificate/8GR62BCT499V",
      certificateId: "8GR62BCT499V",
      topics: ["Algorithms", "Data Structures", "Problem Solving"],
      logo: "coursera-logo.png"
    },
    {
      title: "Linux and Bash for Data Engineering",
      provider: "Coursera",
      date: "2023",
      description: "Practical course on Linux systems administration and Bash scripting for data engineering workflows.",
      certificateLink: "https://www.coursera.org/account/accomplishments/certificate/CAZUJPW6D4BP",
      certificateId: "CAZUJPW6D4BP",
      topics: ["Linux", "Bash", "Automation"],
      logo: "coursera-logo.png"
    },
    {
      title: "Python and Pandas for Data Engineering",
      provider: "Coursera",
      date: "2023",
      description: "In-depth training on Python programming and Pandas library for data manipulation and analysis in data engineering contexts.",
      certificateLink: "https://www.coursera.org/account/accomplishments/certificate/72QS5JSBC67L",
      certificateId: "72QS5JSBC67L",
      topics: ["Python", "Pandas", "Data Analysis"],
      logo: "coursera-logo.png"
    }
  ];
  
  return (
    <div className="bg-white dark:bg-gray-900 text-gray-900 dark:text-gray-100 min-h-screen">
      
      {/* Hero Section */}
      <motion.section 
        ref={heroRef}
        style={{ opacity: heroOpacity, scale: heroScale }}
        className="relative pt-32 pb-20 md:pt-40 md:pb-32 overflow-hidden"
      >
        {/* Enhanced background with multiple layers */}
        <div className="absolute inset-0 bg-gradient-to-b from-blue-50 via-blue-50/80 to-white dark:from-gray-800/90 dark:via-gray-800/70 dark:to-gray-900 -z-10"></div>
        
        {/* Animated decorative elements */}
        <motion.div 
          className="absolute top-20 right-0 w-80 h-80 rounded-full bg-blue-100/50 dark:bg-blue-900/20 blur-3xl -z-10"
          animate={{ 
            scale: [1, 1.05, 1],
            opacity: [0.5, 0.6, 0.5]
          }}
          transition={{ 
            duration: 8, 
            repeat: Infinity,
            repeatType: "reverse" 
          }}
        ></motion.div>
        
        <motion.div 
          className="absolute -bottom-10 left-10 w-96 h-96 rounded-full bg-indigo-100/40 dark:bg-indigo-900/20 blur-3xl -z-10"
          animate={{ 
            scale: [1, 0.95, 1],
            opacity: [0.4, 0.5, 0.4]
          }}
          transition={{ 
            duration: 10, 
            repeat: Infinity,
            repeatType: "reverse",
            delay: 1
          }}
        ></motion.div>
        
        {/* Additional floating elements */}
        <motion.div 
          className="absolute top-1/2 left-1/4 w-32 h-32 rounded-full bg-green-100/30 dark:bg-green-900/10 blur-2xl -z-10"
          animate={{ 
            y: [-10, 10, -10],
            opacity: [0.3, 0.4, 0.3]
          }}
          transition={{ 
            duration: 12, 
            repeat: Infinity,
            repeatType: "reverse" 
          }}
        ></motion.div>
        
        <motion.div 
          className="absolute bottom-1/3 right-1/4 w-24 h-24 rounded-full bg-purple-100/20 dark:bg-purple-900/10 blur-xl -z-10"
          animate={{ 
            y: [5, -5, 5],
            opacity: [0.2, 0.3, 0.2]
          }}
          transition={{ 
            duration: 15, 
            repeat: Infinity,
            repeatType: "reverse" 
          }}
        ></motion.div>
        
        {/* Subtle geometric patterns */}
        <div className="absolute inset-0 opacity-5 dark:opacity-10 bg-grid-pattern -z-10"></div>
        
        <div className="container mx-auto px-4 md:px-10 lg:px-16 xl:px-20 max-w-7xl">
          <div className="flex flex-col-reverse md:flex-row items-center md:items-start gap-10">
            {/* Content Column */}
            <motion.div 
              initial="hidden"
              animate="visible"
              variants={staggerContainer}
              className="md:w-3/5"
            >
              {/* Enhanced badge */}
              <motion.div variants={fadeInRight} className="mb-6">
                <div className="inline-flex items-center px-4 py-1.5 rounded-full bg-gradient-to-r from-blue-100 to-blue-50 dark:from-blue-900/50 dark:to-blue-800/30 text-blue-800 dark:text-blue-300 text-sm font-medium backdrop-blur-sm border border-blue-200/50 dark:border-blue-700/30 shadow-sm">
                  <Code size={14} className="mr-2" /> About Me
                </div>
              </motion.div>
              
              {/* Enhanced name heading with animated underline */}
              <motion.div variants={fadeInRight} className="relative mb-4">
                <h1 className="text-4xl md:text-6xl font-bold mb-2 leading-tight">
                  <span className="bg-clip-text text-transparent bg-gradient-to-r from-blue-600 via-blue-500 to-indigo-600 dark:from-blue-400 dark:via-blue-300 dark:to-indigo-400">
                    Juan Lara
                  </span>
                </h1>
                <motion.div 
                  className="h-1 w-24 bg-gradient-to-r from-blue-600 to-indigo-600 dark:from-blue-400 dark:to-indigo-400 rounded-full"
                  animate={{ 
                    width: ["0%", "18%", "16%"],
                    opacity: [0, 1, 0.9]
                  }}
                  transition={{ 
                    duration: 2, 
                    delay: 0.5 
                  }}
                />
              </motion.div>
              
              {/* Enhanced subtitle with better styling */}
              <motion.h2 
                variants={fadeInRight}
                className="text-2xl md:text-3xl text-gray-800 dark:text-gray-200 mb-8 font-medium"
              >
                Computer Scientist & Mathematician
              </motion.h2>
              
              {/* Content paragraphs with enhanced styling */}
              <motion.div 
                variants={fadeInRight}
                className="space-y-5 text-gray-700 dark:text-gray-300 leading-relaxed max-w-3xl"
              >
                <p className="text-lg">
                  Passionate <span className="font-medium text-blue-600 dark:text-blue-400">Machine Learning Specialist</span> with a strong background in Computer Science and Mathematics from <span className="font-medium text-blue-600 dark:text-blue-400">Universidad Nacional de Colombia</span>, complemented by applied research experience at <span className="font-medium text-indigo-600 dark:text-indigo-400">Harvard Business School</span>.
                </p>
                
                <div className="py-1 border-l-2 border-blue-500/30 dark:border-blue-700/50 pl-4">
                  <p>
                    Leveraging my interdisciplinary approach, I translate theoretical insights into practical business applications through scalable predictive models, adaptive generative architectures, and end-to-end data pipeline optimization.
                  </p>
                </div>
                
                {/* <p>
                  I am committed to transforming complex data into strategic decisions that drive organizational impact.
                </p> */}
              </motion.div>
            </motion.div>
            
            {/* Enhanced Profile Image with more sophisticated decorative elements */}
            <motion.div 
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
              className="md:w-2/5 flex justify-center"
              whileHover={{ scale: 1.02, transition: { duration: 0.3 } }}
            >
              <div className="relative">
                {/* Animated decorative circles */}
                <motion.div 
                  className="absolute -top-8 -left-8 w-full h-full bg-blue-200/40 dark:bg-blue-800/30 rounded-full z-0"
                  animate={{ 
                    rotate: [12, 15, 12],
                    scale: [1, 1.05, 1]
                  }}
                  transition={{ 
                    duration: 10, 
                    repeat: Infinity,
                    repeatType: "reverse" 
                  }}
                ></motion.div>
                
                <motion.div 
                  className="absolute -bottom-8 -right-8 w-full h-full bg-indigo-200/40 dark:bg-indigo-800/30 rounded-full z-0"
                  animate={{ 
                    rotate: [-12, -10, -12],
                    scale: [1, 0.98, 1]
                  }}
                  transition={{ 
                    duration: 8, 
                    repeat: Infinity,
                    repeatType: "reverse",
                    delay: 1
                  }}
                ></motion.div>
                
                {/* Subtle glow effect */}
                <div className="absolute inset-0 bg-blue-400/10 dark:bg-blue-400/5 blur-3xl rounded-full transform scale-90 z-5"></div>
                
                {/* Enhanced image container with better shadows and border */}
                <div className="relative w-64 h-64 md:w-80 md:h-80 lg:w-96 lg:h-96 rounded-full overflow-hidden border-4 border-white/90 dark:border-gray-800/90 shadow-[0_0_30px_rgba(37,99,235,0.2)] dark:shadow-[0_0_30px_rgba(37,99,235,0.15)] z-10">
                  <img 
                    src={`${process.env.PUBLIC_URL}/images/Profile.JPG`} 
                    alt="Juan Lara" 
                    className="w-full h-full object-cover"
                  />
                  
                  {/* Subtle overlay for better image integration */}
                  <div className="absolute inset-0 bg-gradient-to-t from-blue-900/10 to-transparent dark:from-blue-900/20 mix-blend-overlay"></div>
                </div>
                
                {/* Animated dots decoration */}
                <div className="absolute -bottom-4 -right-4 z-20">
                  <motion.div 
                    className="flex space-x-1.5"
                    animate={{ 
                      y: [0, -3, 0],
                      opacity: [0.7, 1, 0.7]
                    }}
                    transition={{ 
                      duration: 4, 
                      repeat: Infinity,
                      repeatType: "reverse" 
                    }}
                  >
                    {[0, 1, 2].map(i => (
                      <div 
                        key={i}
                        className="w-2 h-2 rounded-full bg-blue-500 dark:bg-blue-400"
                        style={{ 
                          animationDelay: `${i * 0.2}s`,
                          opacity: 1 - (i * 0.2)
                        }}
                      ></div>
                    ))}
                  </motion.div>
                </div>
              </div>
            </motion.div>
          </div>
        </div>
      </motion.section>
      
      {/* Skills Section */}
      <section className="py-16 bg-gray-50 dark:bg-gray-800">
        <div className="container mx-auto px-6">
          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-100px" }}
            variants={staggerContainer}
            className="max-w-4xl mx-auto"
          >
            <motion.h2
              variants={fadeInUp}
              className="text-3xl font-bold mb-2 text-center text-gray-800 dark:text-gray-200"
            >
              Technical Proficiency
            </motion.h2>
            <motion.p
              variants={fadeInUp}
              className="italic text-center mb-10 text-gray-600 dark:text-gray-400"
            >
              Languages, frameworks, and tools with proven experience and impact
            </motion.p>

            <motion.div
              variants={fadeInUp}
              className="grid gap-x-12 gap-y-8 md:grid-cols-2"
            >
              {/* ML & AI Frameworks */}
              <div>
                <h3 className="text-xl font-semibold mb-4 text-gray-800 dark:text-gray-200">
                  ML & AI Frameworks
                </h3>
                <SkillBar
                  name="Generative AI"
                  level={4.5}
                  icon={BrainCircuit}
                  color="blue"
                />
                <SkillBar
                  name="Natural Language Processing"
                  level={4.5}
                  icon={Terminal}
                  color="green"
                />
                <SkillBar
                  name="ML Frameworks"
                  level={4.5}
                  icon={Code}
                  color="indigo"
                />
              </div>

              {/* DevOps & Tools */}
              <div>
                <h3 className="text-xl font-semibold mb-4 text-gray-800 dark:text-gray-200">
                  DevOps & Tools
                </h3>
                <SkillBar
                  name="Version Control"
                  level={4.5}
                  icon={Github}
                  color="orange"
                />
                <SkillBar
                  name="Deployment"
                  level={4}
                  icon={Server}
                  color="red"
                />
                <SkillBar
                  name="Automation & Agentic AI"
                  level={4}
                  icon={Layers}
                  color="teal"
                />
              </div>

              {/* Automation & Agentic AI */}
              <div>
                <h3 className="text-xl font-semibold mb-4 text-gray-800 dark:text-gray-200">
                  Cloud & Agentic AI
                </h3>
                <SkillBar
                  name="Cloud Platforms"
                  level={4}
                  icon={Cloud}
                  color="cyan"
                />
                <SkillBar
                  name="Agent Frameworks"
                  level={4}
                  icon={Layers}
                  color="teal"
                />
              </div>

              {/* Visualization & Web Dev */}
              <div>
                <h3 className="text-xl font-semibold mb-4 text-gray-800 dark:text-gray-200">
                  Visualization & Web Dev
                </h3>
                <SkillBar
                  name="Data Visualization"
                  level={4}
                  icon={BarChart}
                  color="purple"
                />
                <SkillBar
                  name="Interactive Applications"
                  level={4}
                  icon={Globe}
                  color="emerald"
                />
                <SkillBar
                  name="Web Development"
                  level={4}
                  icon={Box}
                  color="blueGray"
                />
              </div>
            </motion.div>
          </motion.div>
        </div>
      </section>

      
      {/* Experience Section */}
      <section className="py-16">
        <div className="container mx-auto px-6">
          <motion.div 
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-100px" }}
            variants={staggerContainer}
            className="max-w-4xl mx-auto"
          >
            <motion.div 
              variants={fadeInUp}
              className="flex items-center justify-center mb-10"
            >
              <div className="w-12 h-12 rounded-full bg-blue-100 dark:bg-blue-900/50 flex items-center justify-center mr-4">
                <Briefcase className="text-blue-600 dark:text-blue-400" size={24} />
              </div>
              <h2 className="text-3xl font-bold text-gray-900 dark:text-gray-100">Professional Experience</h2>
            </motion.div>
            
            <div className="relative">
              <div className="absolute left-1/2 transform -translate-x-1/2 h-full w-0.5 bg-blue-200 dark:bg-blue-800 z-0"></div>
              
              <div className="relative z-10">
                {experiences.map((exp, index) => (
                  <ExperienceCard key={index} {...exp} />
                ))}
              </div>
            </div>
          </motion.div>
        </div>
      </section>
      
      {/* Education Section */}
      <section className="py-16 bg-gray-50 dark:bg-gray-800">
        <div className="container mx-auto px-6">
          <motion.div 
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-100px" }}
            variants={staggerContainer}
            className="max-w-4xl mx-auto"
          >
            <motion.div 
              variants={fadeInUp}
              className="flex items-center justify-center mb-10"
            >
              <div className="w-12 h-12 rounded-full bg-indigo-100 dark:bg-indigo-900/50 flex items-center justify-center mr-4">
                <GraduationCap className="text-indigo-600 dark:text-indigo-400" size={24} />
              </div>
              <h2 className="text-3xl font-bold text-gray-900 dark:text-gray-100">Education</h2>
            </motion.div>
            
            <div>
              {education.map((edu, index) => (
                <EducationCard key={index} {...edu} />
              ))}
            </div>
          </motion.div>
        </div>
      </section>
      
      {/* Awards & Recognition Section */}
      <section className="py-16">
        <div className="container mx-auto px-6">
          <motion.div 
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-100px" }}
            variants={staggerContainer}
            className="max-w-4xl mx-auto"
          >
            <motion.div 
              variants={fadeInUp}
              className="flex items-center justify-center mb-10"
            >
              <div className="w-12 h-12 rounded-full bg-yellow-100 dark:bg-yellow-900/30 flex items-center justify-center mr-4">
                <Award className="text-yellow-600 dark:text-yellow-400" size={24} />
              </div>
              <h2 className="text-3xl font-bold text-gray-900 dark:text-gray-100">Awards & Recognition</h2>
            </motion.div>
            
            <motion.div 
              variants={fadeInUp}
              className="space-y-6"
            >
              <div className="bg-white dark:bg-gray-900 rounded-xl p-6 shadow-md border border-gray-100 dark:border-gray-700">
                <div className="flex items-start">
                  <div className="w-10 h-10 rounded-full bg-yellow-100 dark:bg-yellow-900/30 flex-shrink-0 flex items-center justify-center mr-4">
                    <Award className="text-yellow-600 dark:text-yellow-400" size={18} />
                  </div>
                  <div>
                    <h3 className="text-xl font-bold text-gray-900 dark:text-white">Total Ops Star Employee - LATAM</h3>
                    <div className="text-sm text-blue-600 dark:text-blue-400 mb-2">Ipsos • April 2024</div>
                    <p className="text-gray-700 dark:text-gray-300">
                      Recognized for developing TextInsight, demonstrating exceptional initiative, technical expertise, and commitment to operational excellence across Latin America operations.
                    </p>
                  </div>
                </div>
              </div>
              
              <div className="bg-white dark:bg-gray-900 rounded-xl p-6 shadow-md border border-gray-100 dark:border-gray-700">
                <div className="flex items-start">
                  <div className="w-10 h-10 rounded-full bg-yellow-100 dark:bg-yellow-900/30 flex-shrink-0 flex items-center justify-center mr-4">
                    <Award className="text-yellow-600 dark:text-yellow-400" size={18} />
                  </div>
                  <div>
                    <h3 className="text-xl font-bold text-gray-900 dark:text-white">Best Averages Scholarship</h3>
                    <div className="text-sm text-blue-600 dark:text-blue-400 mb-2">Universidad Nacional de Colombia • 2018-2023</div>
                    <p className="text-gray-700 dark:text-gray-300">
                      Awarded for 10 consecutive semesters to the top 15 students with highest academic performance in the program, maintaining excellence throughout entire academic career.
                    </p>
                  </div>
                </div>
              </div>
            </motion.div>
          </motion.div>
        </div>
      </section>
      
      {/* Courses & Certifications Section */}
      <section className="py-16 bg-gray-50 dark:bg-gray-800">
        <div className="container mx-auto px-6">
          <motion.div 
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-100px" }}
            variants={staggerContainer}
            className="max-w-5xl mx-auto"
          >
            <motion.div 
              variants={fadeInUp}
              className="flex items-center justify-center mb-10"
            >
              <div className="w-12 h-12 rounded-full bg-green-100 dark:bg-green-900/30 flex items-center justify-center mr-4">
                <BookOpen className="text-green-600 dark:text-green-400" size={24} />
              </div>
              <h2 className="text-3xl font-bold text-gray-900 dark:text-gray-100">Additional Training</h2>
            </motion.div>
            
            <motion.div 
              variants={fadeInUp}
              className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-2 gap-6"
            >
              {courses.map((course, index) => (
                <CourseCard key={index} {...course} />
              ))}
            </motion.div>
          </motion.div>
        </div>
      </section>
      
      {/* Languages Section */}
      <section className="py-16">
        <div className="container mx-auto px-6">
          <motion.div 
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-100px" }}
            variants={staggerContainer}
            className="max-w-4xl mx-auto"
          >
            <motion.div 
              variants={fadeInUp}
              className="flex items-center justify-center mb-10"
            >
              <div className="w-12 h-12 rounded-full bg-teal-100 dark:bg-teal-900/30 flex items-center justify-center mr-4">
                <Globe className="text-teal-600 dark:text-teal-400" size={24} />
              </div>
              <h2 className="text-3xl font-bold text-gray-900 dark:text-gray-100">Languages</h2>
            </motion.div>
            
            <motion.div 
              variants={fadeInUp}
              className="bg-white dark:bg-gray-900 rounded-xl p-8 shadow-lg border border-gray-100 dark:border-gray-700"
            >
              <div className="grid md:grid-cols-2 gap-6">
                <div className="flex items-center gap-4">
                  <div className="w-16 h-16 rounded-full bg-teal-100 dark:bg-teal-900/30 flex-shrink-0 flex items-center justify-center">
                    <span className="text-2xl font-bold text-teal-700 dark:text-teal-400">ES</span>
                  </div>
                  <div>
                    <h3 className="text-xl font-bold text-gray-900 dark:text-white">Spanish</h3>
                    <p className="text-teal-600 dark:text-teal-400">Native</p>
                  </div>
                </div>
                
                <div className="flex items-center gap-4">
                  <div className="w-16 h-16 rounded-full bg-teal-100 dark:bg-teal-900/30 flex-shrink-0 flex items-center justify-center">
                    <span className="text-2xl font-bold text-teal-700 dark:text-teal-400">EN</span>
                  </div>
                  <div>
                    <h3 className="text-xl font-bold text-gray-900 dark:text-white">English</h3>
                    <p className="text-teal-600 dark:text-teal-400">Advanced</p>
                  </div>
                </div>
              </div>
            </motion.div>
          </motion.div>
        </div>
      </section>
      
      {/* CTA Section */}
      <section className="py-16 bg-gray-900 dark:bg-gray-950">
        <div className="container mx-auto px-6">
          <motion.div 
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-100px" }}
            variants={staggerContainer}
            className="max-w-4xl mx-auto text-center text-white"
          >
            <motion.h2 
              variants={fadeInUp}
              className="text-3xl md:text-4xl font-bold mb-6"
            >
              Let's Work Together
            </motion.h2>
            
            <motion.p 
              variants={fadeInUp}
              className="text-gray-300 mb-8 max-w-2xl mx-auto"
            >
              I'm always open to discussing new projects, research opportunities, or creative collaborations. If you're looking for a Computer Scientist and Mathematician with expertise in Machine Learning and AI, let's connect!
            </motion.p>
            
            <motion.div 
              variants={fadeInUp}
              className="flex flex-col sm:flex-row items-center justify-center gap-4"
            >
              <a 
                href="mailto:larajuand@outlook.com"
                className="px-8 py-3 bg-blue-500 hover:bg-blue-600 text-white rounded-lg transition-colors flex items-center gap-2 font-medium shadow-lg"
              >
                <Mail size={18} />
                <span>Contact Me</span>
              </a>
              
              <Link 
                to="/documents/CV___EN.pdf"
                target="_blank"
                rel="noreferrer"
                className="px-8 py-3 bg-white text-gray-900 border border-gray-200 rounded-lg hover:bg-gray-100 transition-colors flex items-center gap-2 font-medium shadow-lg"
              >
                <svg 
                  className="w-5 h-5" 
                  viewBox="0 0 24 24" 
                  fill="none" 
                  stroke="currentColor" 
                  strokeWidth="2"
                >
                  <path d="M12 15V3m0 12l-4-4m4 4l4-4M2 17l.621 2.485A2 2 0 0 0 4.561 21h14.878a2 2 0 0 0 1.94-1.515L22 17" />
                </svg>
                <span>Download Resume</span>
              </Link >
            </motion.div>
          </motion.div>
        </div>
      </section>
      
    </div>
  );
}