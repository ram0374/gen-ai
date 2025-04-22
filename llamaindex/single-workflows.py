from llama_index.core.workflow import StartEvent, StopEvent, Workflow, step
import asyncio
class MyWorkflow(Workflow):
    @step
    async def my_step(self, ev: StartEvent) -> StopEvent:
        # do something here
        return StopEvent(result="Hello, world!")


async def main():
    w = MyWorkflow(timeout=10, verbose=True)
    result = await w.run()
    print("result:", result)
    
    
asyncio.run(main())    