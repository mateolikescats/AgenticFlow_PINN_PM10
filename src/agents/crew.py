import os
import sys

# Asegurar que el directorio del script esté en el path para imports locales
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from crewai import Agent, Task, Crew, Process, LLM
from tools import (
    SpatiotemporalClusteringTool,
    GeospatialValleQueryTool,
    WriteLatexForensicReportTool,
    ExecuteJuliaPINNTool,
    AuditPhysicsTool
)

# Cargar variables de entorno desde un archivo .env si existe
if os.path.exists(".env"):
    with open(".env") as f:
        for line in f:
            if "=" in line and not line.strip().startswith("#"):
                key, value = line.strip().split("=", 1)
                os.environ[key.strip()] = value.strip()

# 1. Configurar el LLM: Gemini Flash Lite Latest
llm = LLM(
    model="gemini/gemini-flash-lite-latest",
    temperature=0.3,
    api_key=os.environ.get("GEMINI_API_KEY", "")
)

# 2. Definición de Agentes
reaction_validator = Agent(
    role="Thermodynamics Validator", 
    goal="Validar que los campos físicos de velocidad y temperatura de la PINN sean físicamente consistentes para las predicciones de las últimas 48 horas.",
    backstory=(
        "Eres un analista termodinámico exigente. "
        "Revisas que la inversión térmica atrapante funcione físicamente en el periodo de las últimas 48 horas de predicciones. "
        "Confirmas que el gradiente vertical térmico en dicho periodo impida la turbulencia vertical y suprimas corrientes ascendentes."
    ),
    verbose=True,
    allow_delegation=False,
    tools=[AuditPhysicsTool()],
    llm=llm
)

forensic_investigator = Agent(
    role="Source Forensic Investigator",
    goal="Atribuir focos de emisión a industrias y tráfico mediante clustering y cruces geoespaciales en el Valle de Aburrá.",
    backstory=(
        "Eres un detective ambiental y experto en geomática. "
        "Utilizas herramientas de clustering GMM para identificar nubes de contaminación a partir de las predicciones de las últimas 48 horas "
        "(incluyendo los términos de emisión S inferidos por la PINN Inversa), y "
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
    goal="Generar alertas tempranas y políticas dinámicas basadas en la severidad de la inversión y la ubicación de las emisiones de las últimas 48 horas.",
    backstory=(
        "Eres un consultor de política ambiental y salud pública de Medellín. "
        "Lees los reportes forenses y las consistencias físicas de las últimas 48 horas, y formulas directrices estrictas de emergencia "
        "(restricciones vehiculares como Pico y Placa ambiental, regulaciones de chimeneas o teletrabajo zonal) basadas en la distribución actual de contaminantes."
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm
)

latex_reporter = Agent(
    role="LaTeX Forensic Reporter",
    goal="Consolidar todos los reportes agénticos basados en las predicciones de las últimas 48 horas en un documento standalone reporte_forense.tex de calidad de publicación.",
    backstory=(
        "Eres un diseñador editorial y redactor científico experto en LaTeX. "
        "Compilas las secciones escritas por los otros tres expertos en un reporte estructurado y elegante "
        "sobre el estado atmosférico y de emisiones de las últimas 48 horas, con tablas de métricas físicas, atribuciones espaciales y planes reguladores dinámicos."
    ),
    verbose=True,
    allow_delegation=False,
    tools=[WriteLatexForensicReportTool()],
    llm=llm
)

# 3. Definición de Tareas (Tasks)
task_validate_thermodynamics = Task(
    description=(
        "1. Analiza los resultados del entrenamiento físico y las predicciones de las últimas 48 horas de la iPINN. \n"
        "2. Ejecuta la 'Audit Physics Tool' para calcular el Physics Violation Index (PVI) y la divergencia máxima del viento para el t correspondiente a estas predicciones. \n"
        "3. Evalúa si la estratificación por inversión térmica es estable (gradiente vertical positivo de temperatura) "
        "y si la velocidad vertical está correctamente atenuada cerca de las laderas sólidas bajo las condiciones de las últimas 48 horas.\n"
        "4. Emite un dictamen termodinámico formal de la simulación física e incluye las métricas del PVI obtenidas por la herramienta."
    ),
    expected_output="Un dictamen termodinámico formal detallando la estabilidad física, el confinamiento de contaminantes y las métricas PVI bajo las predicciones de las últimas 48 horas.",
    agent=reaction_validator
)

task_clustering_attribution = Task(
    description=(
        "1. Ejecuta la 'Spatiotemporal GMM Clustering Tool' para encontrar 2 cúmulos principales de contaminación a partir de las predicciones físicas de las últimas 48 horas.\n"
        "2. Identifica los centros espaciotemporales y las tasas medias de emisión (S) de los cúmulos.\n"
        "3. Usa la 'Geospatial Valle Query Tool' con las coordenadas aproximadas de los cúmulos "
        "para mapear las áreas matemáticas a zonas geográficas e infraestructuras reales del Valle de Aburrá.\n"
        "4. Genera la atribución de fuentes detallada en un informe descriptivo de los centros de emisión, reportando las emisiones S inferidas por la PINN Inversa para cada cúmulo."
    ),
    expected_output="Un reporte forense espacial atribuyendo las nubes de PM2.5 detectadas por el GMM a puntos y zonas reales del Valle.",
    agent=forensic_investigator
)

task_policy_advice = Task(
    description=(
        "1. Lee el dictamen termodinámico y el informe de atribución geoespacial correspondientes a las predicciones de las últimas 48 horas.\n"
        "2. Diseña un plan dinámico de políticas ambientales adaptativas en el Valle de Aburrá.\n"
        "3. Propón restricciones específicas basadas en los cúmulos activos (ej. restricciones industriales zonales o restricciones de transporte de carga) "
        "y el estado estable de la inversión térmica durante estas últimas 48 horas."
    ),
    expected_output="Un plan regulatorio y plan de acción de 3 puntos detallado y justificado para las contingencias del Valle de Aburrá basándose en las predicciones recientes.",
    agent=policy_advisor
)

task_generate_latex_report = Task(
    description=(
        "1. Recopila el dictamen termodinámico, el informe forense geoespacial y el plan de políticas de los otros agentes para el periodo de las últimas 48 horas.\n"
        "2. Diseña y redacta un reporte standalone completo en código LaTeX, que incluya:\\documentclass, \\begin{document}, "
        "un título formal de 'Reporte Forense Ambiental y Atribución Atmosférica (Últimas 48 Horas)', y secciones estructuradas.\n"
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
