#!/usr/bin/env python3
# -*- coding: utf-8 -*-
######################################################################
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. #
# SPDX-License-Identifier: MIT-0                                #
######################################################################

import json
import random
import ray
from typing import List, Dict
import ast
import pickle
import re
import numpy as np


# Initialize Ray
try:
    ray.init(address='auto', ignore_reinit_error=True)
except:
    ray.init()

# Define company types
COMPANY_TYPES = [
    "OilAndGas",
    "RenewableEnergy",
    "NuclearPower",
    "CoalMining",
    "UtilityProvider"
]


COMPANY_PERSONA = [
    "environmentally conscious",
    "greedy",
    "depressed"    
]

@ray.remote(num_gpus=1)
def call_ollama(prompt):
    from langchain_ollama import OllamaLLM
    
    #https://ollama.com/library/llama3.2:1b
    
    ollama = OllamaLLM(model="llama3.2:3b-instruct-fp16")    
    full_response = ollama.invoke(prompt)        
    return full_response

@ray.remote
class EnergyCompany:
    def __init__(self, company_id: str, company_type: str):
        self.company_id = company_id
        self.company_type = company_type
        self.company_persona =  random.choice(COMPANY_PERSONA)
        self.resources = 1000  # Starting resources
        self.production = 100  # Starting production level
        self.connections: List[str] = []  # Supply chain connections
        self.energy_price = 50 if company_type != "UtilityProvider" else 0  # Price of energy for producers
        self.cost_to_consumer = 20 if company_type == "UtilityProvider" else None  # Cost to consumer for utility providers
        self.production_cost = 30  # Base cost of production
        self.max_capacity = 200  # Maximum production capacity
        self.capacity_upgrade_cost = 50000  # Cost to upgrade capacity by 10 units
        self.capacity_upgrade_amount = 10  # Amount of capacity increase per upgrade
        self.production_cost_penalty = 10  # Increase in maintainance costs
        self.profit = 0
        self.regulator_max = 0.25  # Maximum allowed price for utility providers
        self.connected_companies = {}  # Store data of connected companies
        self.suppliers = {}  # For utility providers: {supplier_id: purchase_amount}
        self.unmet_demand = 0
        self.energy_discount = {}
        
    def get_id(self):
        return self.company_id

    def update_connected_companies(self, connected_data: Dict):
        self.connected_companies = connected_data

    # Choose suppliers based on cost and renewable energy preference
    def choose_suppliers(self, adjust_renewable_mix):
        if self.company_type == "UtilityProvider":
            total_demand = self.production  # Assuming production represents demand for utilities
            available_suppliers = [comp for comp in self.connected_companies.values() if comp['company_type'] != "UtilityProvider"]

            for supplier in available_suppliers:
                if 'energy_price' in supplier:
                    if supplier['company_type'] == "RenewableEnergy":
                        supplier['cost_metric'] = supplier['energy_price'] * ( adjust_renewable_mix + 1)
                    else:
                        supplier['cost_metric'] = supplier['energy_price']
                else:
                    supplier['cost_metric'] = None  # or any default value you prefer
                    
            # Sort suppliers by price
            #sorted_suppliers = sorted(available_suppliers, key=lambda x: x['energy_price'])

            # Sort suppliers by cost metric
            sorted_suppliers = sorted(available_suppliers, key=lambda x: x['cost_metric'])

            self.suppliers = {}
            remaining_demand = total_demand

            # Allocate demand to suppliers in order of cost
            for supplier in sorted_suppliers:
                if remaining_demand <= 0:
                    break
                purchase_amount = min(remaining_demand, supplier['production'])
                self.suppliers[supplier['company_id']] = purchase_amount
                remaining_demand -= purchase_amount

            self.unmet_demand = remaining_demand  # Track unmet demand

    # Update company state based on market conditions and AI decision
    def update_state(self, market_conditions: Dict, subsidies: Dict):

        connected_info = json.dumps(self.connected_companies)
        
        prompt = f"""
        You are an AI assistant simulating the decision-making process of an energy company. Your goal is to maximize profit while considering market conditions, regulations, environmental factors, and your supply chain connections.
        Your persona that drives your decision making on energy sources is: {self.company_persona}

        Company Type: {self.company_type}
        Current Resources: {self.resources}
        Current Production: {self.production}
        Current Energy Price: {self.energy_price}
        Current Cost to Consumer: {self.cost_to_consumer}
        Production Cost: {self.production_cost}
        Maximum Capacity: {self.max_capacity}
        Capacity Upgrade Cost: {self.capacity_upgrade_cost}
        Current Profit: {self.profit}
        Market Conditions: {json.dumps(market_conditions)}
        Subsidies: {json.dumps(subsidies)}
        Connected Companies: {connected_info}
        Current Suppliers: {json.dumps(self.suppliers) if self.company_type == "UtilityProvider" else "N/A"}
        Current Unmet Demand: {self.unmet_demand}

        Based on this information, how should the company adjust its production and pricing strategy?

        Consider the following:
        1. The production and pricing of your connected companies.
        2. The demand in the market and your position in the supply chain.
        3. Available subsidies and how they affect your competitiveness.
        4. Whether upgrading capacity would be beneficial given the current market conditions and your resources.

        For producers:
        1. Adjust production based on market demand, your maximum capacity, and the needs of connected utility providers.
        2. Set energy price considering production costs, market demand, and prices of connected competitors.
        3. Decide whether to upgrade capacity based on market demand and available resources.

        For utility providers:
        1. Adjust production based on market demand, your maximum capacity.
        2. Calculate the average cost of energy from your chosen suppliers.
        3. Set a cost to consumer that covers your expenses and includes a reasonable profit margin.
        4. Ensure the cost to consumer does not exceed the regulator's maximum price of {self.regulator_max * self.production}.
        5. Consider your persona. Set the adjust_renewable_mix value which determines how much renewable energy you choose regardless of price.
           The higher this value the more likely you are to choose renewables. Keep the value between 0 and 10.
        
        Provide your response as a strict JSON object with the following structure, and nothing else. 
        Do not use variables. Do not include comments. 
        {{
            "production_change": <float>,
            "energy_price_change": <float>,
            "cost_to_consumer": <float>,
            "upgrade_capacity": <boolean>,
            "adjust_renewable_mix": <float>
        }}

        Note: Only producers should adjust energy_price and consider upgrading capacity. Only UtilityProvider should set cost_to_consumer.
        """
        # Get AI decision from Ollama
        text = ray.get(call_ollama.remote(prompt))
        
        # Parse the AI decision (JSON string)
        try:
            json_string = clean_llm_output(text)
            decision = json.loads(json_string)
        except ValueError as e:
            print(f"Error parsing JSON: {e}" )
            print("START ERROR TEXT")
            print(text)
            print("END ERROR TEXT")
            
            decision = {
                        "production_change": 0,
                        "energy_price_change": 0,
                        "cost_to_consumer": self.cost_to_consumer or 0,
                        "upgrade_capacity": False,
                        "adjust_renewable_mix" : 0
                    }


        # Convert None to 0 for numerical fields
        for key in ['production_change', 'energy_price_change', 'cost_to_consumer', "adjust_renewable_mix"]:
            if decision.get(key) is None:
                decision[key] = 0

        # Clip renewable mix adjustment to prevent extreme values
        decision["adjust_renewable_mix"] = int(np.clip(decision["adjust_renewable_mix"] , -100,100))

        # Set default value for upgrade_capacity if not present
        if 'upgrade_capacity' not in decision:
            decision['upgrade_capacity'] = False

        if self.company_type != "UtilityProvider":

            # Update company state based on AI decision
            self.production = max(0, min(self.production + decision['production_change'], self.max_capacity))
            self.energy_price = max(self.production_cost, self.energy_price + decision['energy_price_change'])
            
            revenue = self.energy_price * self.production
            costs = self.production_cost * self.production

            # Handle capacity upgrade if decided and resources are available
            if decision['upgrade_capacity'] and self.resources >= self.capacity_upgrade_cost:
                self.resources -= self.capacity_upgrade_cost
                self.max_capacity += self.capacity_upgrade_amount
                self.production_cost += self.production_cost_penalty

                print(f"{self.company_id} upgraded capacity to {self.max_capacity}")

            self.profit = revenue - costs + subsidies.get(self.company_type, 0)

        else:
            # Update state for utility providers
            self.production = market_conditions.get("demand", 0)
            self.choose_suppliers(decision["adjust_renewable_mix"])
            print("suppliers", self.suppliers)

            # Recalculate costs based on new supplier mix            
            try:
                total_energy_cost = sum(self.connected_companies[supplier_id]['energy_price'] * amount * self.energy_discount.get(supplier_id,1.0)
                                        for supplier_id, amount in self.suppliers.items())

            except:
                print("suppliers", self.suppliers)
                print("decision",decision)

            # Apply regulator's maximum price constraint
            max_allowed_cost = self.regulator_max * self.production
            self.cost_to_consumer = min(decision['cost_to_consumer'], max_allowed_cost)

            revenue = self.cost_to_consumer * market_conditions.get("demand", 0)
            costs = total_energy_cost
            self.profit = revenue - costs

            # Calculate and update unmet demand (i.e. "supply" - "demand")
            self.unmet_demand = max(0, self.production - sum(self.suppliers.values()))

        self.resources += self.profit

        # Return updated company state
        return {
            "company_id": self.company_id,
            "company_type": self.company_type,
            "production": self.production,
            "resources": self.resources,
            "energy_price": self.energy_price,
            "cost_to_consumer": self.cost_to_consumer,
            "profit": self.profit,
            "suppliers": self.suppliers if self.company_type == "UtilityProvider" else None,
            "unmet_demand": self.unmet_demand if self.company_type == "UtilityProvider" else None,
            "max_capacity": self.max_capacity,
            "company_persona": self.company_persona,
            "adjust_renewable_mix": decision.get("adjust_renewable_mix",0)
        }

    def add_connection(self, company_id: str):
        self.connections.append(company_id)

    def get_connections(self):
        return self.connections

    def get_current_state(self):
        return {
            "company_id": self.company_id,
            "company_type": self.company_type,
            "production": self.production,
            "resources": self.resources,
            "energy_price": self.energy_price,
            "cost_to_consumer": self.cost_to_consumer,
            "profit": self.profit,
            "suppliers": self.suppliers if self.company_type == "UtilityProvider" else None            
        }

@ray.remote
class EnergyMarket:
    def __init__(self, num_companies: int):
        # Ensure at least one UtilityProvider
        self.companies = [EnergyCompany.remote("Company_0", "UtilityProvider")]
        # Create the rest of the companies
        for i in range(1, num_companies):
            company_type = random.choice(COMPANY_TYPES)
            self.companies.append(EnergyCompany.remote(f"Company_{i}", company_type))

        self.create_supply_chain()
        self.market_conditions = {
            "demand": 1000
        }
        # self.subsidies = {
        #     "RenewableEnergy": 0,
        #     "NuclearPower": 0
        # }
        self.subsidies = {}
        self.company_states = {}  # Store the latest state of each company
        self.demand_history = [self.market_conditions["demand"]]  # Initialize with the starting demand
        self.total_unmet_demand = 0
        
    def create_supply_chain(self):
        """
        Generates a randomized supply chain network for the energy market simulation.
        Each company is assigned a set of random connections to other companies (at least 2, up to max_chain_connections),  
        creating a complex web of potential suppliers and consumers.
        """

        max_chain_connections = 8
        
        company_ids = ray.get([company.get_id.remote() for company in self.companies])
        for company, company_id in zip(self.companies, company_ids):
            num_connections = random.randint(2, max_chain_connections)
            for _ in range(num_connections):
                connection_id = random.choice(company_ids)
                if connection_id != company_id:
                    ray.get(company.add_connection.remote(connection_id))

    def simulate_step(self):
        try:
            # Get current states without calling update_state
            if not self.company_states:  # Only for the first step
                self.company_states = {
                    company_id: ray.get(company.get_current_state.remote())
                    for company_id, company in zip(ray.get([company.get_id.remote() for company in self.companies]), self.companies)
                }

            # Update connected company information for each company
            for company in self.companies:                
                connections = ray.get(company.get_connections.remote())
                connected_data = {conn_id: self.company_states[conn_id] for conn_id in connections if conn_id in self.company_states}
                ray.get(company.update_connected_companies.remote(connected_data))

            # Update states with current market information and connected company data
            results = ray.get([company.update_state.remote(self.market_conditions, self.subsidies) for company in self.companies])

            # Update the stored states
            self.company_states = {result['company_id']: result for result in results}

            self.total_unmet_demand = sum(result['unmet_demand'] for result in results if result['unmet_demand'] is not None)

            return results
        except Exception as e:
            print(f"Error in simulate_step: {str(e)}")
            return []

    def get_total_unmet_demand(self):
      return self.total_unmet_demand

    def update_market_conditions(self):
        # Simulate changes in market conditions
        self.market_conditions["demand"] = max(0, self.market_conditions["demand"] + random.uniform(-50, 50))
        self.demand_history.append(self.market_conditions["demand"])  # Add the new demand to the history

    def update_subsidies(self):
        # Simulate changes in subsidies
        for company_type in COMPANY_TYPES:
            if company_type in self.subsidies:
                self.subsidies[company_type] = max(0, self.subsidies[company_type] + random.uniform(-10, 10))
            # elif random.random() < 0.1:  # 10% chance to introduce new subsidy
            #     self.subsidies[company_type] = random.uniform(0, 100)

    def get_network_data(self):
        company_ids = ray.get([company.get_id.remote() for company in self.companies])
        company_types = ray.get([company.update_state.remote({}, {}) for company in self.companies])
        company_types = {data['company_id']: data['company_type'] for data in company_types}
        connections = ray.get([company.get_connections.remote() for company in self.companies])
        return company_ids, company_types, connections

    def get_subsidies(self):
        return self.subsidies

    def get_demand_history(self):
        return self.demand_history

def save_simulation_data(all_results, all_subsidies, company_data, demand_history, unmet_demand_history):
    # Save all_results
    with open('all_results.pkl', 'wb') as f:
        pickle.dump(all_results, f)

    # Save all_subsidies
    with open('all_subsidies.pkl', 'wb') as f:
        pickle.dump(all_subsidies, f)

    # Save company_data
    with open('company_data.pkl', 'wb') as f:
        pickle.dump(company_data, f)

    # Save demand_history
    with open('demand_history.pkl', 'wb') as f:
        pickle.dump(demand_history, f)

    with open('unmet_demand_history.pkl', 'wb') as f:
        pickle.dump(unmet_demand_history, f)

    print("Simulation data saved to files: all_results.pkl, all_subsidies.pkl, company_data.pkl, demand_history.pkl, and unmet_demand_history.pkl")

def run_simulation(num_companies: int, num_steps: int):
    market = EnergyMarket.remote(num_companies)

    # Visualize initial network
    company_ids, company_types, connections = ray.get(market.get_network_data.remote())

    company_data = {
        'company_ids': company_ids,
        'company_types': company_types,
        'connections': connections
    }

    # Data collection for visualization
    all_results = []
    all_subsidies = []
    unmet_demand_history = []

    for step in range(num_steps):
        print(f"Simulation Step {step + 1}")
        results = ray.get(market.simulate_step.remote())
        subsidies = ray.get(market.get_subsidies.remote())
        all_results.append(results)
        all_subsidies.append(subsidies)
        print(json.dumps(results, indent=2))

        ray.get(market.update_market_conditions.remote())
        ray.get(market.update_subsidies.remote())

        demand_history = ray.get(market.get_demand_history.remote())
        unmet_demand = ray.get(market.get_total_unmet_demand.remote())
        unmet_demand_history.append(unmet_demand)

        # Save all collected data
        save_simulation_data(all_results, all_subsidies, company_data, demand_history, unmet_demand_history)


    return all_results, all_subsidies, demand_history, unmet_demand_history


def safe_eval(expr):
    try:
        return ast.literal_eval(expr)
    except (ValueError, SyntaxError):
        # If literal_eval fails, try a restricted eval
        return eval(expr, {"__builtins__": {}})


def replace_equation(match):
    equation = match.group(1)
    try:
        result = safe_eval(equation)
        return f': {result}'
    except:
        return match.group(0)  # Return the original string if evaluation fails


def clean_llm_output(output):
    # Find the first { and the last }
    start = output.find('{')
    end = output.rfind('}')
    if start != -1 and end != -1:
        json_str = output[start:end+1]
        
        json_str = re.sub(r'//.*', '', json_str)
        
        # Replace equations with their evaluated results
        json_str = re.sub(r':\s*([^,\n}]+)', replace_equation, json_str)
        
        try:
            # Parse JSON
            data = json.loads(json_str)
            # Re-encode to ensure valid JSON
            return json.dumps(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON: {e}")
    else:
        raise ValueError("No valid JSON object found in the output")

              

#%% main
if __name__ == "__main__":
    run_simulation(num_companies=3, num_steps=3)    
    print("Simulation completed. Check the generated PKL files for saved data.")