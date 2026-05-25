import os
from crewai import Agent, Task, Crew, Process, LLM
from tools import SpatiotemporalClusteringTool, GeospatialValleQueryTool, WriteLatexForensicReportTool

# Cargar variables de entorno desde un archivo .env si existe
if os.path.exists(".env"):
    with open(".env") as f:
        for line in f:
            if "=" in line and not line.strip().startswith("#"):
                key, value = line.strip().split("=", 1)
                os.environ[key.strip()] = value.strip()

# 1. Configurar el LLM: Gemini 3.5 Flash
llm = LLM(
    model="gemini/gemini-3.5-flash",
    temperature=0.3,
    api_key=os.environ.get("GEMINI_API_KEY", "")
)

# 2. Definición de Agentes
reaction_validator = Agent(
    role="Thermodynamics Validator",
    goal="Validar que los campos físicos de velocidad y temperatura de la PINN sean físicamente consistentes.",
    backstory=(
        "Eres un analista termodinámico exigente. "
        "Revisas que la inversión térmica atrapante funcione físicamente. "
        "Confirmas que el gradiente vertical térmico impida la turbulencia vertical y suprimas corrientes ascendentes."
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm
)

forensic_investigator = Agent(
    role="Source Forensic Investigator",
    goal="Atribuir focos de emisión a industrias y tráfico mediante clustering y cruces geoespaciales en el Valle de Aburrá.",
    backstory=(
        "Eres un detective ambiental y experto en geomática. "
        "Utilizas herramientas de clustering GMM para identificar nubes de contaminación y "
        "luego consultas bases de datos geoespaciales del Valle de Aburrá para cruzar las coordenadas "
        "matemáticas con autopistas de alto tráfico o zonas industriales pesadas específicas."
    ),
    verbose=True,
    allow_delegation=False,
    tools=[SpatiotemporalClusteringTool(), GeospatialValleQueryTool()],
    llm=llm
)

policy_advisor = Agent(
    role="Environmental Policy Advisor",
    goal="Generar alertas tempranas y políticas dinámicas basadas en la severidad de la inversión y la ubicación de las emisiones.",
    backstory=(
        "Eres un consultor de política ambiental y salud pública de Medellín. "
        "Lees los reportes forenses y las consistencias físicas, y formulas directrices estrictas de emergencia "
        "(restricciones vehiculares como Pico y Placa ambiental, regulaciones de chimeneas o teletrabajo zonal)."
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm
)

latex_reporter = Agent(
    role="LaTeX Forensic Reporter",
    goal="Consolidar todos los reportes agénticos en un documento standalone reporte_forense.tex de calidad de publicación.",
    backstory=(
        "Eres un diseñador editorial y redactor científico experto en LaTeX. "
        "Compilas las secciones escritas por los otros tres expertos en un reporte estructurado y elegante "
        "con tablas de métricas físicas, atribuciones espaciales y planes reguladores dinámicos."
    ),
    verbose=True,
    allow_delegation=False,
    tools=[WriteLatexForensicReportTool()],
    llm=llm
)

# 3. Definición de Tareas (Tasks)
task_validate_thermodynamics = Task(
    description=(
        "1. Analiza los resultados del entrenamiento físico de la iPINN en laderas parabólicas. \n"
        "2. Evalúa si la estratificación por inversión térmica es estable (gradiente vertical positivo de temperatura) "
        "y si la velocidad vertical está correctamente atenuada cerca de las laderas sólidas.\n"
        "3. Emite un dictamen termodinámico formal de la simulación física."
    ),
    expected_output="Un dictamen termodinámico formal detallando la estabilidad física y el confinamiento de contaminantes.",
    agent=reaction_validator
)

task_clustering_attribution = Task(
    description=(
        "1. Ejecuta la 'Spatiotemporal GMM Clustering Tool' para encontrar 2 cúmulos principales de contaminación.\n"
        "2. Identifica los centros espaciotemporales de los cúmulos.\n"
        "3. Usa la 'Geospatial Valle Query Tool' con las coordenadas aproximadas de los cúmulos "
        "para mapear las áreas matemáticas a zonas geográficas e infraestructuras reales del Valle de Aburrá.\n"
        "4. Genera la atribución de fuentes detallada en un informe descriptivo de los centros de emisión."
    ),
    expected_output="Un reporte forense espacial atribuyendo las nubes de PM2.5 detectadas por el GMM a puntos y zonas reales del Valle.",
    agent=forensic_investigator
)

task_policy_advice = Task(
    description=(
        "1. Lee el dictamen termodinámico y el informe de atribución geoespacial.\n"
        "2. Diseña un plan dinámico de políticas ambientales adaptativas en el Valle de Aburrá.\n"
        "3. Propón restricciones específicas basadas en los cúmulos activos (ej. restricciones industriales zonales o restricciones de transporte público de carga) "
        "y el estado estable de la inversión térmica."
    ),
    expected_output="Un plan regulatorio y plan de acción de 3 puntos detallado y justificado para las contingencias del Valle de Aburrá.",
    agent=policy_advisor
)

task_generate_latex_report = Task(
    description=(
        "1. Recopila el dictamen termodinámico, el informe forense geoespacial y el plan de políticas de los otros agentes.\n"
        "2. Diseña y redacta un reporte standalone completo en código LaTeX, que incluya:\\documentclass, \\begin{document}, "
        "un título formal de 'Reporte Forense Ambiental y Atribución Atmosférica', y secciones estructuradas.\n"
        "3. Usa la 'Write Latex Forensic Report Tool' para escribir el código LaTeX resultante en 'reporte/reporte_forense.tex'."
    ),
    expected_output="Un mensaje de éxito confirmando la escritura del reporte forense en LaTeX.",
    agent=latex_reporter
)

# 4. Ensamblar la Tripulación (Crew)
aburra_crew = Crew(
    agents=[reaction_validator, forensic_investigator, policy_advisor, latex_reporter],
    tasks=[task_validate_thermodynamics, task_clustering_attribution, task_policy_advice, task_generate_latex_report],
    process=Process.sequential,
    verbose=True
)

if __name__ == "__main__":
    print("Iniciando orquestación Multi-Agente de Intérpretes Forenses...")
    if not os.environ.get("GEMINI_API_KEY"):
        print("ERROR: La variable de entorno GEMINI_API_KEY no está configurada.")
        exit(1)
        
    result = aburra_crew.kickoff()
    print("######################")
    print("### REPORTE FINAL ###")
    print("######################")
    print(result)
