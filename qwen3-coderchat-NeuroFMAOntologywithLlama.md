## Qwen3-coder Session - Python Owlready Code for Using Ollama Llama as a Front End for Searching and Interrogating neurofma_fma3.0.owl

### USER
Suggest python code with owlready for using local Ollama Llama as a front end for searching and interrogating the OWL ontology neurofma_fma3.0.owl

### ASSISTANT
Here's a Python solution using owlready2 to load the ontology and interact with a local Ollama Llama model for querying:

```python
import os
from owlready2 import *
from ollama import Client
import json

# Initialize Ollama client
client = Client(host='http://localhost:11434')

# Load the ontology
ontology_path = "neurofma_fma3.0.owl"
onto = get_ontology(f"file://{os.path.abspath(ontology_path)}").load()

# Get all classes in the ontology
classes = list(onto.classes())
class_names = [cls.name for cls in classes]

# Prepare ontology context for LLM
def get_ontology_context():
    context = []
    for cls in classes:
        # Get class information
        class_info = {
            "name": cls.name,
            "label": cls.label[0] if hasattr(cls, 'label') and cls.label else cls.name,
            "description": cls.comment[0] if hasattr(cls, 'comment') and cls.comment else "No description",
            "superclasses": [super_cls.name for super_cls in cls.is_a if super_cls != cls],
            "subclasses": [sub_cls.name for sub_cls in cls.subclasses() if sub_cls != cls]
        }
        context.append(class_info)
    return json.dumps(context, indent=2)

# Query the LLM with ontology context
def query_ontology(query):
    context = get_ontology_context()
    prompt = f"""
    You are an expert in the FMA (Foundational Model of Anatomy) ontology.
    You have access to the following classes and their relationships:
    
    {context}
    
    Query: {query}
    
    Please answer the query using only the information in the ontology.
    If the query cannot be answered with the provided information, say "I cannot answer this query with the provided ontology information."
    """
    
    try:
        response = client.generate(
            model="llama3",
            prompt=prompt,
            options={
                "temperature": 0.2,
                "num_predict": 1000
            }
        )
        return response['response']
    except Exception as e:
        return f"Error querying LLM: {str(e)}"

# Interactive query function
def interactive_query():
    print("NeuroFMA FMA3.0 Ontology Query Interface")
    print("Type 'quit' to exit")
    print("Available classes:", ", ".join(class_names[:10]), "..." if len(class_names) > 10 else "")
    print("-" * 50)
    
    while True:
        query = input("\nEnter your query: ").strip()
        if query.lower() in ['quit', 'exit']:
            break
        if not query:
            continue
            
        print("\nProcessing query...")
        answer = query_ontology(query)
        print(f"\nAnswer: {answer}")

# Example usage
if __name__ == "__main__":
    # Example queries
    example_queries = [
        "What are the main anatomical structures in the brain?",
        "List all subclasses of 'Brain'",
        "What is the relationship between 'Cerebral Cortex' and 'Gray Matter'?",
        "Find all classes related to 'Nervous System'"
    ]
    
    # Run example queries
    for i, query in enumerate(example_queries, 1):
        print(f"\n--- Example {i}: {query} ---")
        answer = query_ontology(query)
        print(answer)
    
    # Interactive mode
    interactive_query()
```

### Requirements:
1. Install required packages:
```bash
pip install owlready2 ollama
```

2. Ensure Ollama is running locally with Llama3 model:
```bash
ollama pull llama3
```

### Features:
1. **Ontology Loading**: Loads the neurofma_fma3.0.owl file
2. **Context Preparation**: Extracts class information for LLM context
3. **LLM Integration**: Uses Ollama's Llama3 model for queries
4. **Interactive Mode**: Allows real-time querying
5. **Example Queries**: Demonstrates functionality with sample questions

### Usage:
1. Place `neurofma_fma3.0.owl` in the same directory
2. Run the script
3. Enter queries about the ontology (e.g., "What are the brain structures?")
4. Use `quit` to exit

### Notes:
- The script extracts class names, labels, descriptions, and relationships
- LLM responses are limited to information in the ontology
- Adjust `num_predict` and `temperature` in `options` for different response styles
- For large ontologies, consider pagination or filtering of context

This implementation provides a bridge between the structured ontology data and natural language processing capabilities of the Llama model, allowing for intuitive querying of complex anatomical relationships.

### USER
How can I extend this to include property relationships (like 'part_of' or 'located_in') in the ontology context?

### ASSISTANT
Here's an enhanced version that includes property relationships in the ontology context:

```python
import os
from owlready2 import *
from ollama import Client
import json

# Initialize Ollama client
client = Client(host='http://localhost:11434')

# Load the ontology
ontology_path = "neurofma_fma3.0.owl"
onto = get_ontology(f"file://{os.path.abspath(ontology_path)}").load()

# Get all classes in the ontology
classes = list(onto.classes())
class_names = [cls.name for cls in classes]

# Get all object properties
object_properties = list(onto.object_properties())
data_properties = list(onto.data_properties())

# Prepare comprehensive ontology context including relationships
def get_ontology_context():
    context = {
        "classes": [],
        "object_properties": [],
        "data_properties": [],
        "class_relationships": [],
        "property_relationships": []
    }
    
    # Process classes
    for cls in classes:
        class_info = {
            "name": cls.name,
            "label": cls.label[0] if hasattr(cls, 'label') and cls.label else cls.name,
            "description": cls.comment[0] if hasattr(cls, 'comment') and cls.comment else "No description",
            "superclasses": [super_cls.name for super_cls in cls.is_a if super_cls != cls],
            "subclasses": [sub_cls.name for sub_cls in cls.subclasses() if sub_cls != cls],
            "equivalent_classes": [eq_cls.name for eq_cls in cls.equivalent_to if eq_cls != cls] if hasattr(cls, 'equivalent_to') else [],
            "disjoint_with": [disj_cls.name for disj_cls in cls.disjoint_with] if hasattr(cls, 'disjoint_with') else []
        }
        context["classes"].append(class_info)
    
    # Process object properties
    for prop in object_properties:
        prop_info = {
            "name": prop.name,
            "label": prop.label[0] if hasattr(prop, 'label') and prop.label else prop.name,
            "description": prop.comment[0] if hasattr(prop, 'comment') and prop.comment else "No description",
            "domain": [domain.name for domain in prop.domain] if prop.domain else [],
            "range": [rng.name for rng in prop.range] if prop.range else [],
            "subproperties": [sub_prop.name for sub_prop in prop.subproperties()] if hasattr(prop, 'subproperties') else [],
            "superproperties": [super_prop.name for super_prop in prop.superproperties()] if hasattr(prop, 'superproperties') else []
        }
        context["object_properties"].append(prop_info)
    
    # Process data properties
    for prop in data_properties:
        prop_info = {
            "name": prop.name,
            "label": prop.label[0] if hasattr(prop, 'label') and prop.label else prop.name,
            "description": prop.comment[0] if hasattr(prop, 'comment') and prop.comment else "No description",
            "domain": [domain.name for domain in prop.domain] if prop.domain else [],
            "range": str(prop.range) if prop.range else "No range"
        }
        context["data_properties"].append(prop_info)
    
    # Extract class relationships (instances of object properties)
    class_relationships = []
    for cls in classes:
        # Get all object property assertions for this class
        for prop in object_properties:
            try:
                # Get all individuals that have this property with this class as domain
                if hasattr(prop, 'domain') and cls in prop.domain:
                    # Get instances of this property
                    instances = list(prop(cls))
                    for instance in instances:
                        if hasattr(instance, 'name'):
                            relationship = {
                                "source": cls.name,
                                "property": prop.name,
                                "target": instance.name,
                                "property_label": prop.label[0] if hasattr(prop, 'label') and prop.label else prop.name
                            }
                            class_relationships.append(relationship)
            except:
                continue
    
    context["class_relationships"] = class_relationships
    
    # Extract property relationships (subproperties, superproperties, etc.)
    property_relationships = []
    for prop in object_properties:
        # Get subproperties
        subprops = prop.subproperties()
        for subprop in subprops:
            property_relationships.append({
                "source": prop.name,
                "relationship": "subproperty_of",
                "target": subprop.name
            })
        
        # Get superproperties
        superprops = prop.superproperties()
        for superprop in superprops:
            property_relationships.append({
                "source": prop.name,
                "relationship": "superproperty_of",
                "target": superprop.name
            })
    
    context["property_relationships"] = property_relationships
    
    return json.dumps(context, indent=2)

# Enhanced query function with relationship-aware prompting
def query_ontology(query):
    context = get_ontology_context()
    
    # Create a more comprehensive prompt that includes relationships
    prompt = f"""
    You are an expert in the FMA (Foundational Model of Anatomy) ontology.
    You have access to the following ontology structure including classes, properties, and relationships:
    
    {context}
    
    Query: {query}
    
    Please answer the query using only the information in the ontology.
    If the query cannot be answered with the provided information, say "I cannot answer this query with the provided ontology information."
    
    When answering:
    1. Reference specific class names, property names, and their relationships
    2. Explain how properties like 'part_of', 'located_in', etc. connect classes
    3. If a query asks about a specific relationship, identify which property defines it
    4. Use the class hierarchy and property domains/ranges to provide accurate answers
    
    Example response format:
    "The class 'Cerebral Cortex' is a subclass of 'Gray Matter' and is part_of the 'Cerebrum' through the 'part_of' property."
    """
    
    try:
        response = client.generate(
            model="llama3",
            prompt=prompt,
            options={
                "temperature": 0.2,
                "num_predict": 1500,
                "top_k": 50,
                "top_p": 0.9
            }
        )
        return response['response']
    except Exception as e:
        return f"Error querying LLM: {str(e)}"

# Function to get specific relationship information
def get_relationship_info():
    """Extract and display key relationship patterns in the ontology"""
    relationships = []
    
    # Find common anatomical relationships
    common_properties = ['part_of', 'located_in', 'connected_to', 'innervated_by']
    
    for prop_name in common_properties:
        prop = onto.search_one(name=prop_name)
        if prop:
            relationships.append({
                "property": prop_name,
                "description": prop.comment[0] if hasattr(prop, 'comment') and prop.comment else "No description",
                "domain": [domain.name for domain in prop.domain] if prop.domain else [],
                "range": [rng.name for rng in prop.range] if prop.range else []
            })
    
    return relationships

# Interactive query function with relationship awareness
def interactive_query():
    print("NeuroFMA FMA3.0 Ontology Query Interface (with Relationships)")
    print("Type 'quit' to exit")
    print("Available classes:", ", ".join(class_names[:10]), "..." if len(class_names) > 10 else "")
    print("-" * 50)
    
    # Show relationship information
    rel_info = get_relationship_info()
    print("\nCommon Relationships in Ontology:")
    for rel in rel_info:
        print(f"  {rel['property']}: {rel['description']}")
    
    while True:
        query = input("\nEnter your query: ").strip()
        if query.lower() in ['quit', 'exit']:
            break
        if not query:
            continue
            
        print("\nProcessing query...")
        answer = query_ontology(query)
        print(f"\nAnswer: {answer}")

# Example usage with relationship-focused queries
def run_example_queries():
    example_queries = [
        "What anatomical structures are part_of the brain?",
        "How are 'Cerebral Cortex' and 'Cerebrum' related?",
        "What properties define the relationship between brain regions?",
        "List all classes that are located_in the 'Cerebral Cortex'",
        "Show me the relationship hierarchy for 'part_of' property",
        "What is the difference between 'part_of' and 'located_in' properties?",
        "Find all subclasses of 'Gray Matter' and their relationships"
    ]
    
    print("\n" + "="*60)
    print("EXAMPLE QUERIES WITH RELATIONSHIP FOCUS")
    print("="*60)
    
    for i, query in enumerate(example_queries, 1):
        print(f"\n--- Example {i}: {query} ---")
        answer = query_ontology(query)
        print(answer)
        print("-" * 50)

# Function to export relationship statistics
def export_relationship_statistics():
    """Generate statistics about ontology relationships"""
    stats = {
        "total_classes": len(classes),
        "total_object_properties": len(object_properties),
        "total_data_properties": len(data_properties),
        "class_relationships": 0,
        "property_relationships": 0
    }
    
    # Count class relationships
    for cls in classes:
        for prop in object_properties:
            try:
                instances = list(prop(cls))
                stats["class_relationships"] += len(instances)
            except:
                continue
    
    # Count property relationships
    for prop in object_properties:
        stats["property_relationships"] += len(prop.subproperties()) + len(prop.superproperties())
    
    return stats

if __name__ == "__main__":
    # Show statistics
    stats = export_relationship_statistics()
    print("Ontology Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*60)
    
    # Run example queries
    run_example_queries()
    
    # Interactive mode
    interactive_query()
```

### Key Enhancements:

1. **Comprehensive Property Handling**:
   - Object properties (like `part_of`, `located_in`)
   - Data properties (for textual attributes)
   - Property domains and ranges
   - Property hierarchies (subproperties, superproperties)

2. **Detailed Relationship Context**:
   - Class-to-class relationships through object properties
   - Property inheritance relationships
   - Domain and range information for properties
   - Superclass/subclass relationships

3. **Enhanced Prompting**:
   - More specific instructions for handling relationships
   - Clear guidance on how to reference properties and their connections
   - Examples of expected response formats

4. **Additional Features**:
   - Relationship statistics export
   - Common relationship identification
   - Specialized example queries focused on relationships
   - Better error handling for property access

### Usage Examples:

The enhanced version can now answer queries like:
- "What anatomical structures are part_of the brain?"
- "How are 'Cerebral Cortex' and 'Cerebrum' related?"
- "Show me the relationship hierarchy for 'part_of' property"
- "What is the difference between 'part_of' and 'located_in' properties?"

### Requirements:
Same as before, plus:
```bash
pip install owlready2 ollama
```

This extended version provides much richer context for the LLM to understand and reason about the complex relationships within the FMA ontology.

### USER
Could you add support for reasoning about individuals (instances) in addition to classes and properties?

### ASSISTANT
Here's an enhanced version that includes support for reasoning about individuals (instances) in addition to classes and properties:

```python
import os
from owlready2 import *
from ollama import Client
import json

# Initialize Ollama client
client = Client(host='http://localhost:11434')

# Load the ontology
ontology_path = "neurofma_fma3.0.owl"
onto = get_ontology(f"file://{os.path.abspath(ontology_path)}").load()

# Get all classes in the ontology
classes = list(onto.classes())
class_names = [cls.name for cls in classes]

# Get all object properties
object_properties = list(onto.object_properties())
data_properties = list(onto.data_properties())

# Get all individuals (instances)
individuals = list(onto.individuals())
individual_names = [ind.name for ind in individuals]

# Prepare comprehensive ontology context including individuals
def get_ontology_context():
    context = {
        "classes": [],
        "object_properties": [],
        "data_properties": [],
        "individuals": [],
        "class_relationships": [],
        "individual_relationships": [],
        "property_relationships": [],
        "class_instance_mapping": {}
    }
    
    # Process classes
    for cls in classes:
        class_info = {
            "name": cls.name,
            "label": cls.label[0] if hasattr(cls, 'label') and cls.label else cls.name,
            "description": cls.comment[0] if hasattr(cls, 'comment') and cls.comment else "No description",
            "superclasses": [super_cls.name for super_cls in cls.is_a if super_cls != cls],
            "subclasses": [sub_cls.name for sub_cls in cls.subclasses() if sub_cls != cls],
            "equivalent_classes": [eq_cls.name for eq_cls in cls.equivalent_to if eq_cls != cls] if hasattr(cls, 'equivalent_to') else [],
            "disjoint_with": [disj_cls.name for disj_cls in cls.disjoint_with] if hasattr(cls, 'disjoint_with') else []
        }
        context["classes"].append(class_info)
    
    # Process object properties
    for prop in object_properties:
        prop_info = {
            "name": prop.name,
            "label": prop.label[0] if hasattr(prop, 'label') and prop.label else prop.name,
            "description": prop.comment[0] if hasattr(prop, 'comment') and prop.comment else "No description",
            "domain": [domain.name for domain in prop.domain] if prop.domain else [],
            "range": [rng.name for rng in prop.range] if prop.range else []
        }
        context["object_properties"].append(prop_info)
    
    # Process data properties
    for prop in data_properties:
        prop_info = {
            "name": prop.name,
            "label": prop.label[0] if hasattr(prop, 'label') and prop.label else prop.name,
            "description": prop.comment[0] if hasattr(prop, 'comment') and prop.comment else "No description",
            "domain": [domain.name for domain in prop.domain] if prop.domain else [],
            "range": str(prop.range) if prop.range else "No range"
        }
        context["data_properties"].append(prop_info)
    
    # Process individuals
    for ind in individuals:
        ind_info = {
            "name": ind.name,
            "label": ind.label[0] if hasattr(ind, 'label') and ind.label else ind.name,
            "type": [cls.name for cls in ind.is_a] if hasattr(ind, 'is_a') else [],
            "properties": {}
        }
        
        # Collect object property assertions
        object_assertions = []
        for prop in object_properties:
            try:
                values = list(ind.get_relations(prop))
                if values:
                    object_assertions.append({
                        "property": prop.name,
                        "values": [str(val) for val in values]
                    })
            except:
                continue
        
        # Collect data property assertions
        data_assertions = []
        for prop in data_properties:
            try:
                values = list(ind.get_data_properties(prop))
                if values:
                    data_assertions.append({
                        "property": prop.name,
                        "values": [str(val) for val in values]
                    })
            except:
                continue
        
        ind_info["object_properties"] = object_assertions
        ind_info["data_properties"] = data_assertions
        context["individuals"].append(ind_info)
    
    # Build class-instance mapping for quick reference
    class_instance_mapping = {}
    for cls in classes:
        class_instances = []
        for ind in individuals:
            if cls in ind.is_a:
                class_instances.append(ind.name)
        if class_instances:
            class_instance_mapping[cls.name] = class_instances
    context["class_instance_mapping"] = class_instance_mapping
    
    # Process class relationships (subclass relationships)
    class_relationships = []
    for cls in classes:
        for super_cls in cls.is_a:
            if super_cls != cls:
                class_relationships.append({
                    "subject": cls.name,
                    "predicate": "subClassOf",
                    "object": super_cls.name
                })
    context["class_relationships"] = class_relationships
    
    # Process individual relationships
    individual_relationships = []
    for ind in individuals:
        # Get all object property assertions for this individual
        for prop in object_properties:
            try:
                values = list(ind.get_relations(prop))
                for value in values:
                    individual_relationships.append({
                        "subject": ind.name,
                        "predicate": prop.name,
                        "object": str(value)
                    })
            except:
                continue
    context["individual_relationships"] = individual_relationships
    
    # Process property relationships
    property_relationships = []
    for prop in object_properties:
        # Subproperty relationships
        for sub_prop in prop.subproperties():
            property_relationships.append({
                "subject": sub_prop.name,
                "predicate": "subPropertyOf",
                "object": prop.name
            })
        
        # Superproperty relationships
        for super_prop in prop.superproperties():
            property_relationships.append({
                "subject": prop.name,
                "predicate": "subPropertyOf",
                "object": super_prop.name
            })
    
    context["property_relationships"] = property_relationships
    
    return json.dumps(context, indent=2)

# Enhanced query function with individual-aware reasoning
def query_ontology(query):
    context = get_ontology_context()
    
    # Create a more comprehensive prompt that includes individuals
    prompt = f"""
    You are an expert in the FMA (Foundational Model of Anatomy) ontology.
    You have access to the complete ontology structure including classes, properties, and individuals (instances).
    
    Ontology Structure:
    {context}
    
    Query: {query}
    
    Please answer the query using only the information in the ontology.
    
    When answering questions about individuals:
    1. Reference specific individual names and their types/classes
    2. Explain how individuals relate to classes through type assertions
    3. Describe object property relationships between individuals
    4. Mention data property values for individuals when relevant
    
    When answering questions about relationships:
    1. Explain how individuals instantiate classes
    2. Describe how individuals are connected through object properties
    3. Show how data properties apply to individuals
    4. Use the class hierarchy to explain broader relationships
    
    Examples of expected responses:
    "The individual 'Homo_sapiens' is an instance of the class 'Species' and has the property 'taxonomic_rank' with value 'species'."
    "The individual 'Cerebral_Cortex' is part_of the 'Cerebrum' through the 'part_of' property."
    "The individual 'Basal_Ganglia' is located_in the 'Cerebrum' and is connected_to the 'Thalamus' through the 'connected_to' property."
    
    Important: Only use information that exists in the provided ontology structure.
    """
    
    try:
        response = client.generate(
            model="llama3",
            prompt=prompt,
            options={
                "temperature": 0.3,
                "num_predict": 2000,
                "top_k": 50,
                "top_p": 0.9,
                "repeat_penalty": 1.1
            }
        )
        return response['response']
    except Exception as e:
        return f"Error querying LLM: {str(e)}"

# Function to get specific individual information
def get_individual_info(individual_name):
    """Get detailed information about a specific individual"""
    individual = onto.search_one(name=individual_name)
    if not individual:
        return f"Individual '{individual_name}' not found in ontology."
    
    info = {
        "name": individual_name,
        "type": [cls.name for cls in individual.is_a],
        "object_properties": {},
        "data_properties": {}
    }
    
    # Get object property assertions
    for prop in object_properties:
        try:
            values = list(individual.get_relations(prop))
            if values:
                info["object_properties"][prop.name] = [str(val) for val in values]
        except:
            continue
    
    # Get data property assertions
    for prop in data_properties:
        try:
            values = list(individual.get_data_properties(prop))
            if values:
                info["data_properties"][prop.name] = [str(val) for val in values]
        except:
            continue
    
    return info

# Function to find individuals of a specific class
def find_individuals_of_class(class_name):
    """Find all individuals that are instances of a specific class"""
    cls = onto.search_one(name=class_name)
    if not cls:
        return f"Class '{class_name}' not found in ontology."
    
    individuals_of_class = []
    for ind in individuals:
        if cls in ind.is_a:
            individuals_of_class.append(ind.name)
    
    return individuals_of_class

# Interactive query function with individual awareness
def interactive_query():
    print("NeuroFMA FMA3.0 Ontology Query Interface (with Individuals)")
    print("Type 'quit' to exit")
    print("Type 'info <individual_name>' for individual details")
    print("Type 'find <class_name>' to find individuals of a class")
    print("-" * 60)
    
    # Show some statistics
    print(f"Ontology Statistics:")
    print(f"  Classes: {len(classes)}")
    print(f"  Properties: {len(object_properties) + len(data_properties)}")
    print(f"  Individuals: {len(individuals)}")
    
    while True:
        query = input("\nEnter your query: ").strip()
        if query.lower() in ['quit', 'exit']:
            break
        if not query:
            continue
            
        # Handle special commands
        if query.startswith('info '):
            individual_name = query[5:].strip()
            info = get_individual_info(individual_name)
            print(f"\nIndividual Information for '{individual_name}':")
            print(json.dumps(info, indent=2))
            continue
            
        elif query.startswith('find '):
            class_name = query[5:].strip()
            individuals = find_individuals_of_class(class_name)
            print(f"\nIndividuals of class '{class_name}':")
            if isinstance(individuals, list):
                for ind in individuals:
                    print(f"  - {ind}")
            else:
                print(f"  {individuals}")
            continue
        
        print("\nProcessing query...")
        answer = query_ontology(query)
        print(f"\nAnswer: {answer}")

# Example usage with individual-focused queries
def run_example_queries():
    example_queries = [
        "What are the instances of the class 'Cerebral_Cortex'?",
        "Show me information about the individual 'Cerebrum'",
        "List all individuals that are part_of the 'Brain'",
        "Find all individuals of type 'Gray_Matter'",
        "What properties does the individual 'Homo_sapiens' have?",
        "Show the relationships of the individual 'Basal_Ganglia'",
        "Which individuals are located_in the 'Cerebrum'?",
        "What are the data properties of the individual 'Cerebral_Cortex'?",
        "Find individuals that are connected_to the 'Thalamus'",
        "Show all individuals that are instances of 'Species'"
    ]
    
    print("Running example queries with individual information:")
    print("=" * 60)
    
    for i, query in enumerate(example_queries, 1):
        print(f"\n{i}. Query: {query}")
        print("-" * 40)
        answer = query_ontology(query)
        print(f"Answer: {answer[:200]}...")

# Function to get summary statistics
def get_ontology_summary():
    """Get a summary of the ontology structure"""
    summary = {
        "total_classes": len(classes),
        "total_properties": len(object_properties) + len(data_properties),
        "total_individuals": len(individuals),
        "classes_with_instances": len([cls for cls in classes if any(ind in individuals for ind in cls.instances())]),
        "individuals_by_type": {}
    }
    
    # Count individuals by type
    for cls in classes:
        count = len([ind for ind in individuals if cls in ind.is_a])
        if count > 0:
            summary["individuals_by_type"][cls.name] = count
    
    return summary

if __name__ == "__main__":
    # Run example queries
    run_example_queries()
    
    # Interactive mode
    print("\n" + "=" * 60)
    print("Interactive Mode - Start querying!")
    interactive_query()
else:
    # For importing
    pass
```

This enhanced version now includes:

1. **Individual Support**: Full integration of individuals (instances) into the ontology structure
2. **Property Assertions**: Both object and data property assertions for individuals
3. **Type Inheritance**: Proper handling of class-instance relationships
4. **Query Enhancement**: Improved prompting for individual-aware reasoning
5. **Interactive Features**: 
   - `info <individual_name>` to get individual details
   - `find <class_name>` to find individuals of a specific class
6. **Example Queries**: Comprehensive examples showing individual usage
7. **Summary Statistics**: Ontology structure overview

The system now properly handles individuals, their relationships, and can answer questions about specific instances in the ontology.
```
