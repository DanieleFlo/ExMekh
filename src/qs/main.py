import json
import logging
from pathlib import Path

from dotenv import load_dotenv

from qs.core.agents.deep_agent.deep_agent import DeepAgent
from qs.core.agents.node import Node
from qs.providers.local import LocalProvider

load_dotenv()
logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    provider = LocalProvider(model="qwen3.5-9b-mtp")
    capabilities_paths = [Path(__file__).parent / "capabilities"]
    root_node = Node(
        capabilities_paths=capabilities_paths,
        name_hds="main",
    )
    deep_agent = DeepAgent(provider=provider, node=root_node)
    while True:
        user_message = input("User: ")
        if user_message.lower() in ["exit", "quit"]:
            print("Exiting...")
            break
        response = deep_agent.run(user_message=user_message)
        print("Response:", response.get_message())
        print("history:", json.dumps(response.history, indent=2, ensure_ascii=True))
        print(
            "Deep skill history:\n",
            json.dumps(response.get_history_tree(), indent=2, ensure_ascii=True),
        )


if __name__ == "__main__":
    main()
