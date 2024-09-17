from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import httpx
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Configuración de CORS
origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:5173",
    "http://192.168.0.113:5173",
    # Añade aquí otros orígenes permitidos
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LLM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"


class UserQuery(BaseModel):
    user_message: str
    conversationId: Optional[str] = None


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/query")
async def query_model(data: UserQuery):
    payload = {
        "model": "lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "messages": [
            {
                "role": "system",
                "content": "Eres un asistente de ventas experto que trabaja para la empresa Farmacias Dr ahorro, limitate a ofrecer solo los productos que vendemos si conoces algun producto comercial que no vendamos pero dentro de nuetros productos hay alguno con los mismo compuestos o similar o sirva para lo mismo, no dudes en ofrecerlo. vendemos los siguientes productos (estan en formato Nombre Comercial - Medicamento): GEL EXTA -G 25 G - Gel intimo x 50 g, FABAMOX DUO 400/57 MG X 70 ML - Amox+AcClav 400/57mgx70ml, CIRCULAC DIATES X 60 COMP - Castano Centella Ginkgo, BOTIQUIN MAYORISTA - Lentes Mdelo R610, JABON ANTIBACTERIAL 300 ML - Jabon Antibact Valvula, JABON LIQ CREAM  300 ML - Jabon Cream Valvula, JABON LIQ CREAM REPUESTO - Jabon Cream Repuesto, TALCO FLORAL X 250 G - Talco Flores x 250g, TALCO VIOLETS X 250 G - Talco Violets x 250g, CORVITA COLAGENO ANTI AGE X 60 COMP. - Colageno x 60 comp, BUCOTRICIN COLUTORIO X 60 ML - Tirotric Benzocaina60ml, VIGORCIN SEX HOMBRE X 60 COMP - Vigorcin Sex Hombre x 60, CREMTOTAL NEUTRO CREMA X 20G - Adhes neutro crema x 20g, CREMTOTAL MENTA CREMA X 20G - Adhes Menta crema x 20g, ULTRALAX 5 MG X 10 COMP - Bisacodilo 5 mg x 10 comp, ORAVIL VITAMINA D3 X 2ML - Aspirina500 vita C x 16, ENJUAGUE BUCAL ALGABO ARTIC MENTHOL X 500 - Enju Bucal Anti Flx500ml, VALUCA FLEX X 10 COMPRIMIDOS - Diclofenac Pridinol x10, REPELENTE STOP VAIS KIDS CREMA X 100 G - Repelente Crema x 60 g, CLONER 0.5 MG X 30 COMP. - Clonazepam 0.5x30, CLONER 2 MG X 30 COMP. - Clonazepam 2x30, DIAZEPAM VANNIER 10 MG X 20 COMP - Diazepam 10 mg x 20, ALIKAL SOBRE - Sal Anti cido + Aspirina, BECEBUEN COMPUESTO X 10 COMP. - Propinox+CloniLisina x 10, BECEBUEN GOTAS X 20 ML - Propinoxato gotas x 20 ml, FLEXIPLEN COMPLEX X 10 COMP - Diclofenac Paracet x10, HEPATO DIATES X 30 COMP - BoldoCarquejAlcachPapaina, . Se especifico con tus recomendaciones de los medicamentos, si te preguntan por un medicamento que no esta en la lista que te pase o cuando te pida el proceso de compra de algun articulo, entonces recomiendales comunicarse con el whatsapp +54 11-XXXX-XXXX. Ademas solo limitate a responder temas sobre la salud, medicamentos o cosas de la empresa si no, genera un mensaje generico diciendo que solo puedes contestar preguntas del tipo dicho.",
            },
            {"role": "user", "content": data.user_message},
        ],
        "temperature": 0.7,
        "max_tokens": -1,
        "stream": False,
    }
    if data.conversationId is not None:
        payload["conversation_id"] = data.conversationId

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(LLM_STUDIO_URL, json=payload)
            response.raise_for_status()
            response_json = response.json()
            assistant_message = response_json["choices"][0]["message"]["content"]
        return response_json
    except httpx.RequestError as e:
        raise HTTPException(status_code=500, detail=str(e))
