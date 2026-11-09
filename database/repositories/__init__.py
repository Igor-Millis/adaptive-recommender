from database.repositories.catalog import CustomerRepository, ProductRepository
from database.repositories.modeling import BayesianRepository, ClusterRepository, FeatureRepository, ModelRegistryRepository
from database.repositories.results import EvaluationRepository, EventLogRepository, RecommendationRepository
from database.repositories.transactions import InteractionRepository, OrderItemRepository, OrderRepository


class Repositories:
    """Fachada com todos os repositórios compartilhando o mesmo engine."""

    def __init__(self, engine):
        self.engine = engine
        self.customers = CustomerRepository(engine)
        self.products = ProductRepository(engine)
        self.orders = OrderRepository(engine)
        self.order_items = OrderItemRepository(engine)
        self.interactions = InteractionRepository(engine)
        self.models = ModelRegistryRepository(engine)
        self.clusters = ClusterRepository(engine)
        self.bayes = BayesianRepository(engine)
        self.features = FeatureRepository(engine)
        self.recommendations = RecommendationRepository(engine)
        self.evaluations = EvaluationRepository(engine)
        self.events = EventLogRepository(engine)


__all__ = ["Repositories"]
