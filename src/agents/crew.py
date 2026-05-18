import os
from crewai import Agent, Task, Crew, Process
from langchain_google_genai import ChatGoogleGenerativeAI
from tools import ExecuteJuliaPINNTool, SpatiotemporalClusteringTool

# 1. Configurar el LLM: Gemini 1.5 Pro (o Flash)
# Se requiere que el usuario haya seteado la variable de entorno GEMINI_API_KEY
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro-latest",
    verbose=True,
    temperature=0.3,
    google_api_key=os.environ.get("GEMINI_API_KEY", "")
)

# 2. Definición de Agentes
physics_architect = Agent(
    role="Physics Architect & Computacional Fluid Dynamics Expert",
    goal="Configurar y entrenar el modelo numérico de PINN para resolver la Ecuación de Boussinesq de dispersión de PM2.5.",
    backstory=(
        "Eres un experto en simulación matemática y dinámica de fluidos en valles. "
        "Comprendes cómo las inversiones térmicas atrapan las partículas. "
        "Tu misión es usar tus herramientas para ejecutar el motor numérico en Julia "
        "y evaluar los hiperparámetros (learning_rate y epochs) hasta asegurar la convergencia."
    ),
    verbose=True,
    allow_delegation=False,
    tools=[ExecuteJuliaPINNTool()],
    llm=llm
)

reaction_validator = Agent(
    role="Thermodynamics Validator",
    goal="Verificar que los resultados numéricos no violen las leyes de la termodinámica para la inversión térmica.",
    backstory=(
        "Eres un analista termodinámico exigente. "
        "Revisas los registros del modelo físico. Sabes que si la inversión térmica "
        "ocurre, el aire frío debe estar abajo y atrapar el PM2.5. Validarás el razonamiento del Architect."
    ),
    verbose=True,
    allow_delegation=True,
    llm=llm
)

forensic_investigator = Agent(
    role="Source Forensic Investigator",
    goal="Atribuir la contaminación a fuentes específicas mediante clustering espaciotemporal.",
    backstory=(
        "Eres un detective medioambiental. Usas algoritmos de Machine Learning (como GMM) "
        "para rastrear las nubes móviles de contaminación en el espacio y en el tiempo, "
        "descubriendo cómo se unen o separan."
    ),
    verbose=True,
    allow_delegation=False,
    tools=[SpatiotemporalClusteringTool()],
    llm=llm
)

# 3. Definición de Tareas (Tasks)
task_train_pinn = Task(
    description=(
        "1. Usa la 'Execute Julia PINN Tool' con epochs=50 y learning_rate=0.01.\n"
        "2. Lee los logs de compilación devueltos. \n"
        "3. Si el Loss final es muy alto (>0.5), vuelve a correr la herramienta "
        "con learning_rate más bajo o más epochs.\n"
        "4. Genera un reporte sobre la convergencia y entregalo."
    ),
    expected_output="Un reporte detallado del proceso de convergencia de la red neuronal en Julia, incluyendo el Loss final.",
    agent=physics_architect
)

task_validate_thermodynamics = Task(
    description=(
        "1. Revisa el reporte de convergencia entregado por el Physics Architect.\n"
        "2. Comprueba lógicamente si el modelo parece estable y apto para ser usado "
        "en las atribuciones de PM2.5.\n"
        "3. Escribe un visto bueno termodinámico o solicita re-entrenamiento."
    ),
    expected_output="Un reporte de validación termodinámica (Pass/Fail) con justificación física.",
    agent=reaction_validator
)

task_clustering_attribution = Task(
    description=(
        "1. Usa la 'Spatiotemporal GMM Clustering Tool' indicando buscar 2 componentes (nubes).\n"
        "2. Analiza el reporte de cúmulos devuelto por la herramienta.\n"
        "3. Combina este análisis con el reporte de validación termodinámica.\n"
        "4. Elabora el informe forense final que asigne la posible procedencia "
        "de las nubes de PM2.5 a fuentes específicas."
    ),
    expected_output="Un informe forense de 3 párrafos atribuyendo la contaminación del Valle de Aburrá a cúmulos dinámicos específicos.",
    agent=forensic_investigator
)

# 4. Ensamblar la Tripulación (Crew)
aburra_crew = Crew(
    agents=[physics_architect, reaction_validator, forensic_investigator],
    tasks=[task_train_pinn, task_validate_thermodynamics, task_clustering_attribution],
    process=Process.sequential,
    verbose=True
)

if __name__ == "__main__":
    print("Iniciando orquestación Multi-Agente con Gemini 1.5 Pro...")
    # Requiere GEMINI_API_KEY en el entorno
    if not os.environ.get("GEMINI_API_KEY"):
        print("ERROR: La variable de entorno GEMINI_API_KEY no está configurada.")
        print("Por favor, configúrala antes de ejecutar este script.")
        exit(1)
        
    result = aburra_crew.kickoff()
    print("######################")
    print("### REPORTE FINAL ###")
    print("######################")
    print(result)
