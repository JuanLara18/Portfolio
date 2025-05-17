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
  Brain, 
  LineChart, 
  Globe,
  Terminal, 
  Mail 
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
  
  return (
    <div className="mb-6" ref={ref}>
      <div className="flex items-center mb-2">
        <div className={`w-8 h-8 rounded-md bg-${color}-100 dark:bg-${color}-900/30 flex items-center justify-center mr-3`}>
          <Icon size={18} className={`text-${color}-600 dark:text-${color}-400`} />
        </div>
        <span className="text-gray-800 dark:text-gray-200 font-medium">{name}</span>
      </div>
      <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
        <motion.div 
          className={`h-full bg-${color}-600 dark:bg-${color}-500 rounded-full`}
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
  const heroOpacity = useTransform(scrollY, [0, 300], [1, 0.6]);
  const heroScale = useTransform(scrollY, [0, 300], [1, 0.95]);
  
  // Experience data
  const experiences = [
    {
      role: "Research Assistant",
      company: "Harvard Business School",
      period: "Sep 2022 - Present",
      location: "Boston, USA (Remote)",
      description: "Conduct advanced research in organizational economics with Professor Jorge Tamayo, focusing on theoretical modeling and computational implementation.",
      responsibilities: [
        "Formalized organizational hierarchies and resource optimization through economic frameworks, creating rigorous mathematical proofs to analyze policy impacts on productivity.",
        "Translated theoretical concepts into actionable insights by developing ML models (XGBoost, clustering), implementing NLP pipelines, and building modular data architectures with visualization techniques."
      ],
      skills: ["Machine Learning", "NLP", "Data Architecture", "Visualization", "XGBoost"],
      logo: "hbs-logo.png"
    },
    {
      role: "Data Scientist",
      company: "Ipsos",
      period: "Feb 2024 - Jan 2025",
      location: "Bogota, D.C., Colombia (Hybrid)",
      description: "Transformed market research operations through strategic data science implementations, earning recognition as Total Ops Star Employee for LATAM.",
      responsibilities: [
        "Developed a mobile R Shiny application with embedded ML models for pharmacy segmentation, integrating geospatial mapping (Leaflet) with cloud storage, reducing manual classification work by 85%.",
        "Created TextInsight, a comprehensive Python library using transformer models (BERT) and network visualization for multilingual survey analysis, reducing processing time by 60%.",
        "Designed dynamic dashboards and visualization tools that streamlined decision-making processes across Latin American operations."
      ],
      skills: ["R Shiny", "Leaflet", "BERT", "Cloud Storage", "NetworkX"],
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
        <div className="absolute inset-0 bg-gradient-to-b from-blue-50 to-white dark:from-gray-800 dark:to-gray-900 -z-10"></div>
        
        {/* Decorative elements */}
        <div className="absolute top-20 right-0 w-64 h-64 rounded-full bg-blue-100/50 dark:bg-blue-900/20 blur-3xl -z-10"></div>
        <div className="absolute bottom-10 left-10 w-72 h-72 rounded-full bg-indigo-100/30 dark:bg-indigo-900/10 blur-3xl -z-10"></div>
        
        <div className="container mx-auto px-6">
          <div className="flex flex-col-reverse md:flex-row items-center md:items-start gap-10">
            <motion.div 
              initial="hidden"
              animate="visible"
              variants={staggerContainer}
              className="md:w-2/3"
            >
              <motion.div variants={fadeInRight} className="mb-4">
                <div className="inline-flex items-center px-3 py-1 rounded-full bg-blue-100 text-blue-800 dark:bg-blue-900/50 dark:text-blue-300 text-sm font-medium mb-4">
                  <Code size={14} className="mr-1.5" /> About Me
                </div>
              </motion.div>
              
              <motion.h1 
                variants={fadeInRight}
                className="text-4xl md:text-5xl font-bold mb-6 leading-tight"
              >
                <span className="bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-indigo-600 dark:from-blue-400 dark:to-indigo-400">
                  Juan Lara
                </span>
              </motion.h1>
              
              <motion.h2 
                variants={fadeInRight}
                className="text-2xl md:text-3xl text-gray-800 dark:text-gray-200 mb-6 font-medium"
              >
                Computer Scientist & Applied Mathematician
              </motion.h2>
              
              <motion.div 
                variants={fadeInRight}
                className="space-y-4 text-gray-700 dark:text-gray-300 leading-relaxed"
              >
                <p>
                  Computer Scientist and Mathematician trained at the Universidad Nacional de Colombia, with specialized expertise in Machine Learning. Currently collaborating on applied research at Harvard Business School, where I merge theoretical rigor with computational solutions to study the structure of firms.
                </p>
                <p>
                  Throughout my career, I have developed advanced mathematical models, implemented numerical simulations, and applied machine learning techniques to transform data into strategic insights. My interdisciplinary approach enables me to turn complex theories into practical and efficient tools that drive real-world impact.
                </p>
                <p>
                  Proactive, creative, and problem-solving-oriented, I excel at collaborating and communicating complex ideas clearly. I am particularly interested in research projects and consulting engagements that integrate mathematical foundations with computational solutions to address challenging organizational and strategic problems.
                </p>
              </motion.div>
            </motion.div>
            
            <motion.div 
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
              className="md:w-1/3 flex justify-center"
            >
              <div className="relative">
                <div className="absolute -top-6 -left-6 w-full h-full bg-blue-200/30 dark:bg-blue-900/20 rounded-full transform rotate-12 z-0"></div>
                <div className="absolute -bottom-6 -right-6 w-full h-full bg-indigo-200/30 dark:bg-indigo-900/20 rounded-full transform -rotate-12 z-0"></div>
                
                <div className="relative w-56 h-56 md:w-64 md:h-64 rounded-full overflow-hidden border-4 border-white dark:border-gray-800 shadow-xl z-10">
                  <img 
                    src={`${process.env.PUBLIC_URL}/images/profile.png`} 
                    alt="Juan Lara" 
                    className="w-full h-full object-cover"
                  />
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
              className="text-3xl font-bold mb-10 text-center"
            >
              Technical Expertise
            </motion.h2>
            
            <motion.div 
              variants={fadeInUp}
              className="grid md:grid-cols-2 gap-x-12 gap-y-2"
            >
              <div>
                <h3 className="text-xl font-semibold mb-6 text-gray-800 dark:text-gray-200">Core Competencies</h3>
                <SkillBar name="Machine Learning & AI" level={4.5} icon={Brain} color="blue" />
                <SkillBar name="Mathematical Modeling" level={4.5} icon={LineChart} color="indigo" />
                <SkillBar name="Data Science" level={4} icon={Database} color="purple" />
                <SkillBar name="Natural Language Processing" level={4} icon={Terminal} color="green" />
              </div>
              
              <div>
                <h3 className="text-xl font-semibold mb-6 text-gray-800 dark:text-gray-200">Technical Proficiency</h3>
                <SkillBar name="Python" level={4.5} icon={Code} color="yellow" />
                <SkillBar name="R & SQL" level={4} icon={Server} color="red" />
                <SkillBar name="Interactive Applications" level={4.2} icon={Globe} color="teal" />
                <SkillBar name="Research & Problem Solving" level={4.3} icon={BookOpen} color="orange" />
              </div>
            </motion.div>
            
            <motion.div 
              variants={fadeInUp}
              className="mt-16 bg-white dark:bg-gray-900 rounded-xl p-8 shadow-lg border border-gray-100 dark:border-gray-700"
            >
              <h3 className="text-xl font-semibold mb-6 text-center text-gray-800 dark:text-gray-200">Area of Research Interest</h3>
              
              <div className="grid md:grid-cols-2 gap-6">
                <div className="bg-blue-50 dark:bg-blue-900/30 p-6 rounded-lg">
                  <h4 className="text-lg font-medium mb-3 text-blue-700 dark:text-blue-300">AI Agents & Multi-agent Systems</h4>
                  <p className="text-gray-700 dark:text-gray-300">
                    Exploring the development and optimization of autonomous AI agents and their interactions within multi-agent environments, with applications in organizational decision-making and strategic simulations.
                  </p>
                </div>
                
                <div className="bg-indigo-50 dark:bg-indigo-900/30 p-6 rounded-lg">
                  <h4 className="text-lg font-medium mb-3 text-indigo-700 dark:text-indigo-300">Computational Organizational Theory</h4>
                  <p className="text-gray-700 dark:text-gray-300">
                    Investigating mathematical models of organizational structure, knowledge flows, and decision hierarchies using computational simulations and empirical validation.
                  </p>
                </div>
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