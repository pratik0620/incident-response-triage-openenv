from server.incident_response_triage_env_environment import IncidentResponseTriageEnvironment
from models import IncidentResponseTriageAction

env = IncidentResponseTriageEnvironment()
obs = env.reset()

action = IncidentResponseTriageAction(
    action_type="read_logs",
    reasoning="check logs",
    answer="db issue"
)

obs = env.step(action)
print(obs.done, obs.reward)