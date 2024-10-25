package main

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"github.com/alexflint/go-arg"
)

const OLLAMA_HOST = "http://192.168.1.230:11434"

type OllamaMessage struct {
	Role    string
	Content string
	Images  []string
}

type OllamaRequest struct {
	Model    string
	Stream   bool
	Messages []OllamaMessage
	Format   string
	Options  []string
}

type OllamaResponse struct {
	Message OllamaMessage `json:"message"`
}

func GenerateWithImage(prompt, imagePath string) (string, error) {
	// First, convert the image to base64
	imageData, err := os.ReadFile(imagePath)
	if err != nil {
		return "", fmt.Errorf("failed to read image: %w", err)
	}

	base64Image := base64.StdEncoding.EncodeToString(imageData)

	msg := OllamaMessage{
		Role:    "user",
		Content: prompt,
		Images:  []string{base64Image},
	}

	// Prepare the request body
	reqBody := OllamaRequest{
		Model:    "x/llama3.2-vision",
		Messages: []OllamaMessage{msg},
		Stream:   false,
	}

	// Marshal the request body to JSON
	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return "", fmt.Errorf("failed to marshal JSON: %w", err)
	}

	// Create a new request
	req, err := http.NewRequest(
		"POST",
		OLLAMA_HOST+"/api/chat",
		bytes.NewBuffer(jsonData),
	)
	if err != nil {
		return "", fmt.Errorf("failed to create request: %w", err)
	}

	// Set content-type header
	req.Header.Set("Content-Type", "application/json")

	// Make the request
	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("failed to make request: %w", err)
	}
	defer resp.Body.Close()

	// Read the response body
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("failed to read response: %w", err)
	}

	// Check status code
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("unexpected status code: %d, body: %s", resp.StatusCode, string(body))
	}

	// Parse the response
	var response OllamaResponse
	if err := json.Unmarshal(body, &response); err != nil {
		return "", fmt.Errorf("failed to parse response: %w", err)
	}
	return response.Message.Content, nil
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

type Args struct {
	Path          string `arg:"positional,required" help:"Path to an image or a directory with images"`
	WriteCaptions bool   `arg:"--write-caption,-w" help:"Write captions as .txt (stripping the original extension)"`
	StartCaption  string `arg:"--start,-s" help:"Start the caption with this (image of Leela the dog,)"`
	EnddCaption   string `arg:"--end,-e" help:"End the caption with this (in the style of 'something')"`
	Prompt        string `arg:"--prompt,-p" help:"The prompt to use" default:"Please describe the content and style of this image in detail. Answer only with one sentence that is starting with \"A ...\""`
}

func main() {
	var args Args

	arg.MustParse(&args)

	//  and mention "colorized photo"
	err := ProcessImages(args.Path, func(path string, root string) {
		captionText, err := GenerateWithImage(args.Prompt, path)
		if err != nil {
			log.Fatalf("Aborting because of %v", err)
		}
		captionText = args.StartCaption + captionText + args.EnddCaption
		fmt.Printf("%s: %s\n", strings.TrimPrefix(path, root), captionText)
		captionFile := strings.TrimSuffix(path, filepath.Ext(path)) + ".txt"
		if args.WriteCaptions {
			err = os.WriteFile(captionFile, []byte(captionText), 0644)
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
