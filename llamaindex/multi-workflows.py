from llama_index.core.workflow import StartEvent, StopEvent, Workflow, step
from llama_index.core.workflow import Event
import asyncio
from llama_index.utils.workflow import draw_all_possible_flows



class ProcessingEvent(Event):
    intermediate_result: str

class MultiStepWorkflow(Workflow):
    @step
    async def step_one(self, ev: StartEvent) -> ProcessingEvent:
        # Process initial data
        return ProcessingEvent(intermediate_result="Step 1 complete")
    @step
    async def step_two(self, ev: ProcessingEvent) -> StopEvent:
        # Use the intermediate result
        final_result = f"Finished processing: {ev.intermediate_result}"
        return StopEvent(result=final_result)

async def main():
    w = MultiStepWorkflow(timeout=10, verbose=True)
    draw_all_possible_flows(w, "flow.html")
    result = await w.run()
    print("result:", result)
    
    
asyncio.run(main())    