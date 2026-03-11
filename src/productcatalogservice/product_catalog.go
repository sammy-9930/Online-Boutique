package main

import (
	"context"
	"time"

	pb "github.com/GoogleCloudPlatform/microservices-demo/src/productcatalogservice/genproto"
	"google.golang.org/grpc/codes"
	healthpb "google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/status"
)

type productCatalog struct {
	pb.UnimplementedProductCatalogServiceServer
	catalog pb.ListProductsResponse
	agent   *ProductCatalogAgent
}

func (p *productCatalog) Check(ctx context.Context, req *healthpb.HealthCheckRequest) (*healthpb.HealthCheckResponse, error) {
	return &healthpb.HealthCheckResponse{Status: healthpb.HealthCheckResponse_SERVING}, nil
}

func (p *productCatalog) Watch(req *healthpb.HealthCheckRequest, ws healthpb.Health_WatchServer) error {
	return status.Errorf(codes.Unimplemented, "health check via Watch not implemented")
}

func (p *productCatalog) ListProducts(ctx context.Context, req *pb.Empty) (*pb.ListProductsResponse, error) {
	start := time.Now()
	time.Sleep(extraLatency)
	log.Printf("ListProducts called")

	products := p.parseCatalog()
	results, err := p.agent.RunListProducts(ctx, products)
	if err != nil {
		log.Printf("ListProducts agent failed: %v", err)
		return nil, status.Errorf(codes.Internal, "failed to list products")
	}

	log.Printf("ListProducts returning %d products total_latency_ms=%d",
		len(results), time.Since(start).Milliseconds())

	return &pb.ListProductsResponse{Products: results}, nil
}

func (p *productCatalog) GetProduct(ctx context.Context, req *pb.GetProductRequest) (*pb.Product, error) {
	start := time.Now()
	time.Sleep(extraLatency)
	log.Printf("GetProduct called with id=%s", req.Id)

	products := p.parseCatalog()
	result, err := p.agent.RunGetProduct(ctx, products, req.Id)
	if err != nil {
		log.Printf("GetProduct agent failed: %v", err)
		return nil, status.Errorf(codes.Internal, "failed to get product")
	}

	if result == nil {
		return nil, status.Errorf(codes.NotFound, "no product with ID %s", req.Id)
	}

	log.Printf("GetProduct returning product id=%s name=%s total_latency_ms=%d",
		result.Id, result.Name, time.Since(start).Milliseconds())

	return result, nil
}

func (p *productCatalog) SearchProducts(ctx context.Context, req *pb.SearchProductsRequest) (*pb.SearchProductsResponse, error) {
	start := time.Now()
	time.Sleep(extraLatency)
	log.Printf("SearchProducts called with query=%s", req.Query)

	decision, err := p.agent.DecideAction(req.Query)
	if err != nil {
		log.Printf("AGENT: decision failed: %v", err)
		return &pb.SearchProductsResponse{Results: []*pb.Product{}}, nil
	}

	products := p.parseCatalog()
	log.Printf("CATALOG: loaded %d products", len(products))

	var results []*pb.Product

	switch decision.Tool {
	case "search_by_keyword":
		results = p.agent.SearchByKeyword(products, decision.Query)
	case "search_by_category":
		results = p.agent.SearchByCategory(products, decision.Query)
	case "get_product_by_id":
		results = p.agent.GetProductByID(products, decision.Query)
	default:
		log.Printf("AGENT: unknown tool selected: %s", decision.Tool)
		results = []*pb.Product{}
	}

	log.Printf("SearchProducts returning %d products using tool=%s total_latency_ms=%d",
		len(results), decision.Tool, time.Since(start).Milliseconds())

	return &pb.SearchProductsResponse{Results: results}, nil
}

func (p *productCatalog) parseCatalog() []*pb.Product {
	if reloadCatalog || len(p.catalog.Products) == 0 {
		err := loadCatalog(&p.catalog)
		if err != nil {
			return []*pb.Product{}
		}
	}
	return p.catalog.Products
}