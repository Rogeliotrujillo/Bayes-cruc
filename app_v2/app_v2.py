import reflex as rx
import numpy as np

from sklearn.naive_bayes import GaussianNB

# ============================================
# ENTRENAMIENTO DEL MODELO DE IA (4 variables)
# ============================================
modelo = GaussianNB()

# Datos de entrenamiento: [Asistencia, Estudio, Tareas, Foros]
X_train = np.array([
    # Estudiantes que APRUEBAN (1)
    [10, 10, 10, 10], [9, 9, 9, 9], [8, 8, 8, 8], [7, 7, 8, 7], [6, 6, 7, 6],
    [9, 2, 8, 7], [8, 3, 9, 8], [7, 4, 8, 6], [6, 5, 7, 7],
    [10, 8, 9, 10], [9, 7, 10, 9], [8, 9, 8, 8], [7, 10, 7, 9],
    
    # Estudiantes que REPRUEBAN (0)
    [5, 5, 5, 5], [4, 4, 4, 4], [3, 3, 3, 3], [2, 2, 2, 2], [1, 1, 1, 1],
    [2, 9, 8, 9], [3, 8, 7, 8], [4, 7, 9, 8], [5, 6, 2, 3],
    [6, 5, 1, 2], [7, 4, 2, 1], [8, 3, 1, 9], [1, 10, 9, 10],
])

y_train = np.array([
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  # Aprueban
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0   # Reprueban
])

modelo.fit(X_train, y_train)

# ============================================
# ESTADO DE LA APLICACIÓN
# ============================================
class EstadoAcademico(rx.State):
    # Variables
    asistencia: int = 5
    estudio: int = 5
    tareas: int = 5
    foros: int = 5
    probabilidad_exito: int = 50
    
    # Actualizadores
    def actualizar_asistencia(self, valor):
        self.asistencia = valor[0] if isinstance(valor, list) else valor
        self.calcular_probabilidad()
    
    def actualizar_estudio(self, valor):
        self.estudio = valor[0] if isinstance(valor, list) else valor
        self.calcular_probabilidad()
    
    def actualizar_tareas(self, valor):
        self.tareas = valor[0] if isinstance(valor, list) else valor
        self.calcular_probabilidad()
    
    def actualizar_foros(self, valor):
        self.foros = valor[0] if isinstance(valor, list) else valor
        self.calcular_probabilidad()
    
    # Cálculo de probabilidad con todas las variables
    def calcular_probabilidad(self):
        entrada = np.array([[
            self.asistencia, 
            self.estudio,
            self.tareas,
            self.foros
        ]])
        prob = modelo.predict_proba(entrada)[0][1] * 100
        self.probabilidad_exito = int(prob)

# ============================================
# COMPONENTES REUTILIZABLES
# ============================================

def slider_card(titulo, icono, variable, estado, color, descripcion):
    """Slider con diseño limpio"""
    return rx.card(
        rx.vstack(
            rx.hstack(
                rx.text(icono, size="4"),
                rx.text(titulo, font_weight="bold", size="3"),
                rx.badge(getattr(estado, variable), color_scheme=color),
                justify="between",
                width="100%",
                align="center",
            ),
            rx.text(descripcion, size="1", color="gray"),
            rx.slider(
                on_change=getattr(estado, f"actualizar_{variable}"),
                default_value=[5],
                min=0,
                max=10,
                step=1,
                width="100%",
            ),
            rx.hstack(
                rx.text("0", size="1", color="gray"),
                rx.text("5", size="1", color="gray"),
                rx.text("10", size="1", color="gray"),
                justify="between",
                width="100%",
            ),
            spacing="2",
        ),
        width="100%",
        background="#1e1e1e",
        border_radius="0.5em",
    )

# ============================================
# INTERFAZ PRINCIPAL
# ============================================
def index():
    return rx.container(
        rx.vstack(
            # ========== ENCABEZADO ==========
            rx.center(
                rx.vstack(
                    rx.text("🎓", size="7"),
                    rx.heading("Bayes-CRUC", size="7", text_align="center"),
                    rx.text("Sistema de Inferencia de Riesgo Académico", size="4", color="gray", text_align="center"),
                    rx.text("Basado en el Teorema de Bayes", size="2", color="gray", text_align="center"),
                    rx.divider(width="100%"),
                    rx.text(
                        "Herramienta predictiva que evalúa la probabilidad de éxito académico "
                        "basándose en 4 factores clave.",
                        size="2",
                        color="gray",
                        text_align="center",
                    ),
                    spacing="2",
                    align="center",
                ),
                width="100%",
            ),
            
            # ========== MÉTRICAS ==========
            rx.hstack(
                rx.badge("📊 Naive Bayes", size="2", variant="surface"),
                rx.badge("🎯 4 Variables", size="2", variant="surface"),
                rx.badge("📈 Análisis en tiempo real", size="2", variant="surface"),
                justify="center",
                spacing="3",
                width="100%",
            ),
            
            rx.divider(),
            
            # ========== DATOS DEL ESTUDIANTE ==========
            rx.heading("📝 Factores de Evaluación", size="5"),
            
            # Grid de 2 columnas para los 4 sliders
            rx.grid(
                slider_card(
                    "Asistencia a Clases", "📚", "asistencia", EstadoAcademico, "blue",
                    "Porcentaje de asistencia a clases presenciales"
                ),
                slider_card(
                    "Estudio Autónomo", "📖", "estudio", EstadoAcademico, "green",
                    "Horas dedicadas al estudio independiente"
                ),
                slider_card(
                    "Entrega de Tareas", "📝", "tareas", EstadoAcademico, "purple",
                    "Puntualidad y calidad en entregas (0=tarde, 10=puntual)"
                ),
                slider_card(
                    "Participación en Foros", "💬", "foros", EstadoAcademico, "yellow",
                    "Nivel de participación en foros académicos"
                ),
                columns="2",
                spacing="4",
                width="100%",
            ),
            
            rx.divider(),
            
            # ========== RESULTADOS ==========
            rx.heading("📊 Resultado del Análisis", size="5"),
            
            # Tarjeta de probabilidad principal
            rx.card(
                rx.vstack(
                    rx.cond(
                        EstadoAcademico.probabilidad_exito >= 70,
                        rx.badge("🟢 RIESGO BAJO", color_scheme="green", size="3"),
                        rx.cond(
                            EstadoAcademico.probabilidad_exito >= 40,
                            rx.badge("🟡 RIESGO MODERADO", color_scheme="orange", size="3"),
                            rx.badge("🔴 RIESGO ALTO", color_scheme="red", size="3"),
                        ),
                    ),
                    rx.text(f"{EstadoAcademico.probabilidad_exito}%", size="8", font_weight="bold",
                            color=rx.cond(EstadoAcademico.probabilidad_exito >= 70, "green",
                                         rx.cond(EstadoAcademico.probabilidad_exito >= 40, "orange", "red"))),
                    rx.text("Probabilidad de Éxito Académico", size="2", color="gray"),
                    rx.divider(),
                    rx.cond(
                        EstadoAcademico.probabilidad_exito >= 70,
                        rx.vstack(
                            rx.text("✅ Perfil académico excelente", color="green", size="3"),
                            rx.text("El estudiante cumple con los estándares de éxito", size="2", color="gray", text_align="center"),
                            align="center",
                        ),
                        rx.cond(
                            EstadoAcademico.probabilidad_exito >= 40,
                            rx.vstack(
                                rx.text("⚠️ Perfil con margen de mejora", color="orange", size="3"),
                                rx.text("Se requiere atención en áreas específicas", size="2", color="gray", text_align="center"),
                                align="center",
                            ),
                            rx.vstack(
                                rx.text("🔴 Perfil crítico", color="red", size="3"),
                                rx.text("Intervención inmediata necesaria", size="2", color="gray", text_align="center"),
                                align="center",
                            ),
                        ),
                    ),
                    spacing="3",
                    align="center",
                ),
                width="100%",
                background="#1e1e1e",
                border_radius="0.5em",
            ),
            
            # ========== RECOMENDACIONES COMPACTAS ==========
            rx.card(
                rx.vstack(
                    rx.hstack(
                        rx.text("💡", size="3"),
                        rx.text("Recomendaciones", font_weight="bold", size="3"),
                        spacing="2",
                        align="center",
                    ),
                    rx.divider(),
                    rx.cond(
                        EstadoAcademico.probabilidad_exito >= 70,
                        rx.grid(
                            rx.hstack(rx.text("✅", size="2"), rx.text("Mantener ritmo", size="1"), spacing="1", align="center"),
                            rx.hstack(rx.text("🎯", size="2"), rx.text("Ser tutor", size="1"), spacing="1", align="center"),
                            rx.hstack(rx.text("📈", size="2"), rx.text("Temas avanzados", size="1"), spacing="1", align="center"),
                            rx.hstack(rx.text("🏆", size="2"), rx.text("Postular a becas", size="1"), spacing="1", align="center"),
                            columns="4",
                            spacing="2",
                            width="100%",
                        ),
                        rx.cond(
                            EstadoAcademico.probabilidad_exito >= 40,
                            rx.grid(
                                rx.hstack(rx.text("📅", size="2"), rx.text("Mejorar asistencia", size="1"), spacing="1", align="center"),
                                rx.hstack(rx.text("⏰", size="2"), rx.text("Horario estudio", size="1"), spacing="1", align="center"),
                                rx.hstack(rx.text("🤝", size="2"), rx.text("Grupos estudio", size="1"), spacing="1", align="center"),
                                rx.hstack(rx.text("📝", size="2"), rx.text("Entregas a tiempo", size="1"), spacing="1", align="center"),
                                columns="4",
                                spacing="2",
                                width="100%",
                            ),
                            rx.grid(
                                rx.hstack(rx.text("🚨", size="2"), rx.text("Contactar", size="1"), spacing="1", align="center"),
                                rx.hstack(rx.text("📞", size="2"), rx.text("Consejería", size="1"), spacing="1", align="center"),
                                rx.hstack(rx.text("📖", size="2"), rx.text("Plan recuperación", size="1"), spacing="1", align="center"),
                                rx.hstack(rx.text("👨‍👩‍👧", size="2"), rx.text("Involucrar tutores", size="1"), spacing="1", align="center"),
                                columns="4",
                                spacing="2",
                                width="100%",
                            ),
                        ),
                    ),
                    spacing="2",
                ),
                width="100%",
                background="#1e1e1e",
                border_radius="0.5em",
            ),
            
            # ========== ¿CÓMO FUNCIONA? ==========
            rx.card(
                rx.vstack(
                    rx.hstack(
                        rx.text("❓", size="3"),
                        rx.text("¿Cómo funciona?", font_weight="bold", size="3"),
                        spacing="2",
                        align="center",
                    ),
                    rx.divider(),
                    rx.vstack(
                        rx.hstack(
                            rx.text("1️⃣", size="2"),
                            rx.text("Ingresa los datos del estudiante en los 4 factores de evaluación", size="2", color="gray"),
                            spacing="2",
                            align="center",
                            width="100%",
                        ),
                        rx.hstack(
                            rx.text("2️⃣", size="2"),
                            rx.text("El sistema aplica el Teorema de Bayes para calcular la probabilidad", size="2", color="gray"),
                            spacing="2",
                            align="center",
                            width="100%",
                        ),
                        rx.hstack(
                            rx.text("3️⃣", size="2"),
                            rx.text("La IA actualiza la predicción en tiempo real al modificar los valores", size="2", color="gray"),
                            spacing="2",
                            align="center",
                            width="100%",
                        ),
                        rx.hstack(
                            rx.text("4️⃣", size="2"),
                            rx.text("Se genera un diagnóstico con recomendaciones personalizadas", size="2", color="gray"),
                            spacing="2",
                            align="center",
                            width="100%",
                        ),
                        spacing="2",
                        align="start",
                    ),
                    spacing="2",
                ),
                width="100%",
                background="#1e1e1e",
                border_radius="0.5em",
            ),
            
            # ========== FOOTER ==========
            rx.divider(),
            rx.hstack(
                rx.text("© 2026 Bayes-CRUC | Sistema basado en Teorema de Bayes", size="1", color="gray"),
                rx.text("Naive Bayes • Probabilidad Posterior • Inferencia Bayesiana", size="1", color="gray"),
                justify="between",
                width="100%",
            ),
            
            spacing="5",
            align_items="stretch",
        ),
        max_width="1200px",
        padding="2em",
        background="#0a0a0a",
        min_height="100vh",
    )

# ============================================
# CONFIGURACIÓN DE LA APP
# ============================================
app = rx.App()
app.add_page(index)