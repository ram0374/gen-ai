#pk-lf-fbd29fa6-a566-496f-a807-45c070a06bd8

import os
import base64
from smolagents import CodeAgent, HfApiModel

from opentelemetry.sdk.trace import TracerProvider

from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
LANGFUSE_PUBLIC_KEY="pk-lf-fbd29fa6-a566-496f-a807-45c070a06bd8"
LANGFUSE_SECRET_KEY="sk-lf-89f0da70-8602-4690-8ccf-a7db2d228fc2"
os.environ["HF_TOKEN"]="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"
LANGFUSE_AUTH=base64.b64encode(f"{LANGFUSE_PUBLIC_KEY}:{LANGFUSE_SECRET_KEY}".encode()).decode()

#os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "https://us.cloud.langfuse.com" # EU data region
os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "https://us.cloud.langfuse.com/api/public/otel" # US data region
os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {LANGFUSE_AUTH}"




trace_provider = TracerProvider()
trace_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter()))

SmolagentsInstrumentor().instrument(tracer_provider=trace_provider)


#codeAgent
agent = CodeAgent(tools=[], model=HfApiModel())
alfred_agent = agent.from_hub('RamPasupula/PartyAgent', trust_remote_code=True)
alfred_agent.run("Give me the best playlist for a party at Wayne's mansion. The party idea is a 'villain masquerade' theme") 



#ToolCallingAgent
from smolagents import ToolCallingAgent, DuckDuckGoSearchTool, HfApiModel

agent = ToolCallingAgent(tools=[DuckDuckGoSearchTool()], model=HfApiModel())

agent.run("Search for the best music recommendations for a party at the Wayne's mansion.") 
