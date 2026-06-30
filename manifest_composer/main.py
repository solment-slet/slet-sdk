from slet_sdk.aelite.manifest import AgentManifest
from pydantic_editor import up_server


up_server(AgentManifest, port=8081, agent_manifest_mode=True, accent_colors={'acc':'#e6a800','acc2':'#ff6b35','acc3':'#00c896'})
