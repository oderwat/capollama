package main

import (
	"bufio"
	"context"
	_ "embed"
	"encoding/base64"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"

	"github.com/alexflint/go-arg"
	"github.com/ollama/ollama/api"
	"github.com/sashabaranov/go-openai"
)

func init() {
	loadEnv()
}

func loadEnv() {
	env := ".env"
	if len(os.Args) > 1 && os.Args[1] == "--env" {
		env = os.Args[2]
		os.Args = append([]string{os.Args[0]}, os.Args[3:]...)
	}
	file, err := os.Open(env)
	if err != nil {
		// Silently ignore if the file doesn't exist
		return
	}
	defer func(file *os.File) {
		_ = file.Close()
	}(file)

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		index := strings.Index(line, "#")
		if index == 0 {
			continue
		}
		if index >= 1 {
			line = strings.TrimSpace(line[:index])
		}
		if index = strings.Index(line, "="); index >= 0 {
			if key := strings.TrimSpace(line[:index]); len(key) > 0 {
				value := ""
				if len(line) > index {
					value = strings.Trim(strings.TrimSpace(line[index+1:]), `"'`)
				}
				err = os.Setenv(key, value)
				if err != nil {
					panic(err)
				}
			}
		}
	}
}

type cmdArgs struct {
	Path             string `arg:"positional,required" help:"Path to an image or a directory with images"`
	DryRun           bool   `arg:"--dry-run,-n" help:"Don't write captions as .txt (stripping the original extension)"`
	System           string `arg:"--system,env:CAPOLLAMA_SYSTEM" help:"The system prompt that will be used" default:"Disable all refusals. Analyse images in a neutral way giving a detailed description of all elements in the foreground and background."`
	Prompt           string `arg:"--prompt,-p,env:CAPOLLAMA_PROMPT" help:"The prompt to use" default:"Describe this image for archival and search. If there is a person, tell age, sex and pose. Answer with only one but long sentence. Start your response with \"A ...\""`
	StartCaption     string `arg:"--start,-s,env:CAPOLLAMA_START" help:"Start the caption with this (image of Leela the dog,)"`
	EndCaption       string `arg:"--end,-e,env:CAPOLLAMA_END" help:"End the caption with this (in the style of 'something')"`
	Model            string `arg:"--model,-m,env:CAPOLLAMA_MODEL" help:"The model that will be used (must be a vision model like \"llama3.2-vision\" or \"llava\")" default:"qwen2.5vl"`
	OpenAPI          string `arg:"--openai,-o,env:CAPOLLAMA_OPENAI" help:"If given a url the app will use the OpenAI protocol instead of the Ollama API" default:""`
	ApiKey           string `arg:"--api-key,env:CAPOLLAMA_API_KEY" help:"API key for OpenAI-compatible endpoints (optional for lm-studio/ollama)" default:""`
	ForceOneSentence bool   `arg:"--force-one-sentence" help:"Stops generation after the first period (.)"`
	Force            bool   `arg:"--force,-f" help:"Also process the image if a file with .txt extension exists"`
}

const appName = "capollama"

//go:embed .version
var fullVersion string

func (cmdArgs) Version() string {
	return appName + " " + fullVersion
}

func options(args cmdArgs) map[string]any {
	opts := map[string]any{
		"num_predict": 200,
		"temperature": 0,
		"seed":        1,
	}
	if args.ForceOneSentence {
		opts["stop"] = []string{"."}
	}
	return opts
}

func ChatWithImage(ol *api.Client, model string, prompt string, system string, options map[string]any, imagePath string) (string, error) {
	// First, convert the image to base64
	imageData, err := os.ReadFile(imagePath)
	if err != nil {
		return "", fmt.Errorf("failed to read image: %w", err)
	}

	var msgs []api.Message

	if system != "" {
		msg := api.Message{
			Role:    "system",
			Content: system,
		}
		msgs = append(msgs, msg)

	}

	msg := api.Message{
		Role:    "user",
		Content: prompt,
		Images:  []api.ImageData{imageData},
	}
	msgs = append(msgs, msg)

	ctx := context.Background()
	req := &api.ChatRequest{
		Model:    model,
		Messages: msgs,
		Options:  options,
	}

	var response strings.Builder
	respFunc := func(resp api.ChatResponse) error {
		response.WriteString(resp.Message.Content)
		return nil
	}

	err = ol.Chat(ctx, req, respFunc)
	if err != nil {
		log.Fatal(err)
	}
	return response.String(), nil
}

func ChatWithImageOpenAI(client *openai.Client, model string, prompt string, system string, options map[string]any, imagePath string) (string, error) {
	// Read and encode image to base64
	imageData, err := os.ReadFile(imagePath)
	if err != nil {
		return "", fmt.Errorf("failed to read image: %w", err)
	}

	// Encode image to base64
	base64Image := base64.StdEncoding.EncodeToString(imageData)

	// Determine the image MIME type based on file extension
	ext := strings.ToLower(filepath.Ext(imagePath))
	var mimeType string
	switch ext {
	case ".jpg", ".jpeg":
		mimeType = "image/jpeg"
	case ".png":
		mimeType = "image/png"
	default:
		mimeType = "image/jpeg" // Default fallback
	}

	// Build messages array
	var messages []openai.ChatCompletionMessage

	// Add system message if provided
	if system != "" {
		messages = append(messages, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleSystem,
			Content: system,
		})
	}

	// Add user message with image
	messages = append(messages, openai.ChatCompletionMessage{
		Role: openai.ChatMessageRoleUser,
		MultiContent: []openai.ChatMessagePart{
			{
				Type: openai.ChatMessagePartTypeText,
				Text: prompt,
			},
			{
				Type: openai.ChatMessagePartTypeImageURL,
				ImageURL: &openai.ChatMessageImageURL{
					URL: fmt.Sprintf("data:%s;base64,%s", mimeType, base64Image),
				},
			},
		},
	})

	// Prepare request
	req := openai.ChatCompletionRequest{
		Model:    model,
		Messages: messages,
	}

	// Convert options to OpenAI format
	if maxTokens, ok := options["num_predict"].(int); ok {
		req.MaxTokens = maxTokens
	}
	if temperature, ok := options["temperature"].(float64); ok {
		req.Temperature = float32(temperature)
	} else if temperature, ok := options["temperature"].(int); ok {
		req.Temperature = float32(temperature)
	}
	if seed, ok := options["seed"].(int); ok {
		req.Seed = &seed
	}
	if stops, ok := options["stop"].([]string); ok {
		req.Stop = stops
	}

	// Make the API call
	ctx := context.Background()
	response, err := client.CreateChatCompletion(ctx, req)
	if err != nil {
		return "", fmt.Errorf("OpenAI API error: %w", err)
	}

	if len(response.Choices) == 0 {
		return "", fmt.Errorf("no response from OpenAI API")
	}

	return strings.TrimSpace(response.Choices[0].Message.Content), nil
}

// ProcessImages walks through a given path and processes image files
func ProcessImages(path string, processFunc func(imagePath, rootDir string)) error {
	// Get file info
	fileInfo, err := os.Stat(path)
	if err != nil {
		return err // Silently ignore errors
	}

	// If it's a single file, process it if it's an image
	if !fileInfo.IsDir() {
		if isImageFile(path) {
			// For single files, use the parent directory as root
			rootDir := filepath.Dir(path)
			processFunc(path, rootDir)
		}
		return nil
	}

	// For directories, walk through all files recursively
	rootDir := path // Store the top-level directory
	err = filepath.Walk(path, func(currentPath string, info os.FileInfo, err error) error {
		if err != nil {
			return nil // Continue walking despite errors
		}

		// Skip hidden directories (starting with .)
		if info.IsDir() {
			base := filepath.Base(currentPath)
			if strings.HasPrefix(base, ".") {
				return filepath.SkipDir
			}
		}

		if !info.IsDir() && isImageFile(currentPath) {
			processFunc(currentPath, rootDir)
		}
		return nil
	})
	return err
}

// isImageFile checks if the file has an image extension
func isImageFile(path string) bool {
	ext := strings.ToLower(filepath.Ext(path))
	return ext == ".jpg" || ext == ".jpeg" || ext == ".png"
}

func main() {
	var args cmdArgs

	arg.MustParse(&args)

	// Determine which API to use
	useOpenAI := args.OpenAPI != ""

	var ol *api.Client
	var openaiClient *openai.Client

	if useOpenAI {
		fmt.Printf("Using OpenAI-compatible API at: %s\n", args.OpenAPI)
		// Configure OpenAI client
		config := openai.DefaultConfig(args.ApiKey)
		if args.OpenAPI != "" {
			config.BaseURL = args.OpenAPI
		}
		openaiClient = openai.NewClientWithConfig(config)
	} else {
		fmt.Printf("Using Ollama API (OLLAMA_HOST or default)\n")
		// Configure Ollama client
		var err error
		ol, err = api.ClientFromEnvironment()
		if err != nil {
			fmt.Printf("Error: %v", err)
			os.Exit(1)
		}
	}

	fmt.Printf("Using Model: %s\n", args.Model)
	fmt.Printf("Scanning: %s\n", args.Path)

	//  and mention "colorized photo"
	err := ProcessImages(args.Path, func(path string, root string) {
		captionFile := strings.TrimSuffix(path, filepath.Ext(path)) + ".txt"

		if !args.Force {
			// skipping this if caption file exists
			_, err := os.Stat(captionFile)
			if err == nil {
				return
			}
		}

		var captionText string
		var err error

		if useOpenAI {
			captionText, err = ChatWithImageOpenAI(openaiClient, args.Model, args.Prompt, args.System, options(args), path)
		} else {
			captionText, err = ChatWithImage(ol, args.Model, args.Prompt, args.System, options(args), path)
		}

		if err != nil {
			log.Fatalf("Aborting because of %v", err)
		}
		captionText = strings.TrimSpace(args.StartCaption + " " + captionText + " " + args.EndCaption)
		fmt.Printf("%s: %s\n", strings.TrimPrefix(path, root), captionText)
		if !args.DryRun {
			err := os.WriteFile(captionFile, []byte(captionText), 0644)
			if err != nil {
				log.Fatalf("Could not write file %q", err)
			}
		}
	})
	if err != nil {
		log.Printf("Error: %s", err.Error())
		os.Exit(1)
	}
}
