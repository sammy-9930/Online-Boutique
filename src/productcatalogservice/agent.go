package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"time"

	pb "github.com/GoogleCloudPlatform/microservices-demo/src/productcatalogservice/genproto"
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/ollama"
)

type ProductCatalogAgent struct {
	llm      llms.Model
	llmModel string
}

type AgentDecision struct {
	Tool  string `json:"tool"`
	Query string `json:"query"`
}

func NewProductCatalogAgent() (*ProductCatalogAgent, error) {
	model := os.Getenv("AGENT_MODEL")
	if model == "" {
		model = "llama3"
	}

	baseURL := os.Getenv("OLLAMA_BASE_URL")
	if baseURL == "" {
		baseURL = "http://localhost:11434"
	}

	llmClient, err := ollama.New(
		ollama.WithModel(model),
		ollama.WithServerURL(baseURL),
	)

	if err != nil {
		return nil, fmt.Errorf("failed to create Ollama client: %w", err)
	}

	return &ProductCatalogAgent{
		llm:      llmClient,
		llmModel: model,
	}, nil
}

func cleanLLMJSONResponse(resp string) string {
	resp = strings.TrimSpace(resp)
	resp = strings.TrimPrefix(resp, "```json")
	resp = strings.TrimPrefix(resp, "```")
	resp = strings.TrimSuffix(resp, "```")
	resp = strings.TrimSpace(resp)

	start := strings.Index(resp, "{")
	end := strings.LastIndex(resp, "}")
	if start != -1 && end != -1 && end > start {
		resp = resp[start : end+1]
	}

	return resp
}

// --------------------
// Agent runners
// --------------------

func (a *ProductCatalogAgent) RunListProducts(ctx context.Context, products []*pb.Product) ([]*pb.Product, error) {
	start := time.Now()
	log.Printf("AGENT[ListProducts]: invoked")

	results := a.LoadAllProducts(products)

	log.Printf("AGENT[ListProducts]: returning %d products total_latency_ms=%d",
		len(results), time.Since(start).Milliseconds())

	return results, nil
}

func (a *ProductCatalogAgent) RunGetProduct(ctx context.Context, products []*pb.Product, productID string) (*pb.Product, error) {
	start := time.Now()
	log.Printf("AGENT[GetProduct]: invoked with id=%s", productID)

	result := a.GetSingleProductByID(products, productID)
	if result == nil {
		log.Printf("AGENT[GetProduct]: no product found for id=%s total_latency_ms=%d",
			productID, time.Since(start).Milliseconds())
		return nil, nil
	}

	log.Printf("AGENT[GetProduct]: found product id=%s name=%s total_latency_ms=%d",
		result.Id, result.Name, time.Since(start).Milliseconds())

	return result, nil
}

func (a *ProductCatalogAgent) RunSearchProducts(ctx context.Context, products []*pb.Product, query string) ([]*pb.Product, string, error) {
	start := time.Now()
	log.Printf("AGENT[SearchProducts]: invoked with query=%s", query)

	decision, err := a.DecideAction(query)
	if err != nil {
		log.Printf("AGENT[SearchProducts]: decision failed: %v", err)
		return []*pb.Product{}, "", err
	}

	var results []*pb.Product

	switch decision.Tool {
	case "search_by_keyword":
		results = a.SearchByKeyword(products, decision.Query)
	case "search_by_category":
		results = a.SearchByCategory(products, decision.Query)
	case "get_product_by_id":
		results = a.GetProductByID(products, decision.Query)
	default:
		log.Printf("AGENT[SearchProducts]: unknown tool selected=%s", decision.Tool)
		results = []*pb.Product{}
	}

	log.Printf("AGENT[SearchProducts]: returning %d products using tool=%s total_latency_ms=%d",
		len(results), decision.Tool, time.Since(start).Milliseconds())

	return results, decision.Tool, nil
}

// --------------------
// LLM decision for search only
// --------------------

func (a *ProductCatalogAgent) DecideAction(query string) (*AgentDecision, error) {
	start := time.Now()
	log.Printf("AGENT: received query=%s", query)

	input := map[string]string{
		"user_query": query,
	}

	jsonInput, err := json.Marshal(input)
	if err != nil {
		return nil, err
	}

	prompt := fmt.Sprintf(`You are a product catalog agent.

	You must return ONLY valid JSON.
	Do not add markdown.
	Do not add explanation.

	Allowed tools:
	1. "search_by_keyword" -> use when the user is searching by words in product name or description
	2. "search_by_category" -> use when the user clearly mentions a category like accessories, clothing, kitchen, etc.
	3. "get_product_by_id" -> use when the user gives a specific product id

	Return exactly this JSON shape:
	{
	"tool": "search_by_keyword",
	"query": "cleaned query"
	}

	User request:
	%s`, string(jsonInput))

	resp, err := llms.GenerateFromSinglePrompt(
		context.Background(),
		a.llm,
		prompt,
	)
	if err != nil {
		return nil, err
	}

	log.Printf("AGENT: raw LLM response=%s", resp)

	cleanedResp := cleanLLMJSONResponse(resp)
	log.Printf("AGENT: cleaned LLM response=%s", cleanedResp)

	var decision AgentDecision
	err = json.Unmarshal([]byte(cleanedResp), &decision)
	if err != nil {
		log.Printf("AGENT: failed to parse JSON response: %v", err)
		return nil, err
	}

	if decision.Tool == "" {
		return nil, fmt.Errorf("AGENT: parsed decision has empty tool")
	}

	if decision.Query == "" {
		log.Printf("AGENT: parsed decision has empty query, using original query")
		decision.Query = query
	}

	log.Printf("AGENT: parsed tool=%s query=%s llm_latency_ms=%d",
		decision.Tool, decision.Query, time.Since(start).Milliseconds())

	return &decision, nil
}

// --------------------
// Tools
// --------------------

func (a *ProductCatalogAgent) LoadAllProducts(products []*pb.Product) []*pb.Product {
	start := time.Now()
	log.Printf("TOOL: LoadAllProducts called")

	results := make([]*pb.Product, 0, len(products))
	results = append(results, products...)

	log.Printf("TOOL: LoadAllProducts returning %d products tool_latency_ms=%d",
		len(results), time.Since(start).Milliseconds())

	return results
}

func (a *ProductCatalogAgent) GetSingleProductByID(products []*pb.Product, productID string) *pb.Product {
	start := time.Now()
	log.Printf("TOOL: GetSingleProductByID called with productID=%s", productID)

	for _, product := range products {
		if product.Id == productID {
			log.Printf("TOOL: GetSingleProductByID found product id=%s tool_latency_ms=%d",
				productID, time.Since(start).Milliseconds())
			return product
		}
	}

	log.Printf("TOOL: GetSingleProductByID found 0 products tool_latency_ms=%d",
		time.Since(start).Milliseconds())
	return nil
}

func (a *ProductCatalogAgent) SearchByKeyword(products []*pb.Product, query string) []*pb.Product {
	start := time.Now()
	log.Printf("TOOL: SearchByKeyword called with query=%s", query)

	var results []*pb.Product
	q := strings.ToLower(query)

	for _, product := range products {
		if strings.Contains(strings.ToLower(product.Name), q) ||
			strings.Contains(strings.ToLower(product.Description), q) {
			results = append(results, product)
		}
	}

	log.Printf("TOOL: SearchByKeyword returning %d results tool_latency_ms=%d",
		len(results), time.Since(start).Milliseconds())
	return results
}

func (a *ProductCatalogAgent) SearchByCategory(products []*pb.Product, query string) []*pb.Product {
	start := time.Now()
	log.Printf("TOOL: SearchByCategory called with query=%s", query)

	var results []*pb.Product
	q := strings.ToLower(query)

	for _, product := range products {
		for _, category := range product.Categories {
			if strings.Contains(strings.ToLower(category), q) {
				results = append(results, product)
				break
			}
		}
	}

	log.Printf("TOOL: SearchByCategory returning %d results tool_latency_ms=%d",
		len(results), time.Since(start).Milliseconds())
	return results
}

func (a *ProductCatalogAgent) GetProductByID(products []*pb.Product, productID string) []*pb.Product {
	start := time.Now()
	log.Printf("TOOL: GetProductByID called with productID=%s", productID)

	var results []*pb.Product

	for _, product := range products {
		if product.Id == productID {
			results = append(results, product)
			break
		}
	}

	log.Printf("TOOL: GetProductByID returning %d results tool_latency_ms=%d",
		len(results), time.Since(start).Milliseconds())
	return results
}