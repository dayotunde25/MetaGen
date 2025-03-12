import { useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { useMutation } from "@tanstack/react-query";
import { useToast } from "@/hooks/use-toast";
import { useLocation } from "wouter";
import { 
  Card, 
  CardContent, 
  CardDescription, 
  CardFooter, 
  CardHeader, 
  CardTitle 
} from "@/components/ui/card";
import {
  Form,
  FormControl,
  FormDescription,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Separator } from "@/components/ui/separator";
import { 
  ArrowLeft, 
  Plus, 
  X, 
  Database, 
  Link as LinkIcon, 
  Tag, 
  FileText, 
  HardDrive, 
  Loader2 
} from "lucide-react";
import { insertDatasetSchema, InsertDataset } from "@shared/schema";
import { apiRequest } from "@/lib/queryClient";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Link } from "wouter";

// Extend the schema with additional validation
const addDatasetSchema = insertDatasetSchema.extend({
  url: z.string().url("Please enter a valid URL"),
  name: z.string().min(3, {
    message: "Dataset name must be at least 3 characters",
  }),
  tags: z.array(z.string()).optional(),
});

export default function AddDataset() {
  const { toast } = useToast();
  const [, navigate] = useLocation();
  const [tagInput, setTagInput] = useState("");
  const [validationError, setValidationError] = useState<string | null>(null);

  const form = useForm<z.infer<typeof addDatasetSchema>>({
    resolver: zodResolver(addDatasetSchema),
    defaultValues: {
      name: "",
      description: "",
      url: "",
      source: "",
      format: "",
      size: "",
      tags: [],
    },
  });

  const addDatasetMutation = useMutation({
    mutationFn: async (data: InsertDataset) => {
      const res = await apiRequest("POST", "/api/datasets", data);
      return await res.json();
    },
    onSuccess: (data) => {
      toast({
        title: "Dataset Added",
        description: "Your dataset has been added and queued for processing.",
      });
      navigate(`/dataset/${data.id}`);
    },
    onError: (error) => {
      toast({
        title: "Error Adding Dataset",
        description: error instanceof Error ? error.message : "An unknown error occurred",
        variant: "destructive",
      });
    },
  });

  const onSubmit = (data: z.infer<typeof addDatasetSchema>) => {
    setValidationError(null);
    
    // Ensure tags is an array if it's provided
    if (data.tags === undefined) {
      data.tags = [];
    }
    
    addDatasetMutation.mutate(data);
  };

  const addTag = () => {
    if (!tagInput.trim()) return;
    
    const currentTags = form.getValues("tags") || [];
    if (!currentTags.includes(tagInput.trim())) {
      form.setValue("tags", [...currentTags, tagInput.trim()]);
      setTagInput("");
    }
  };

  const removeTag = (tagToRemove: string) => {
    const currentTags = form.getValues("tags") || [];
    form.setValue(
      "tags",
      currentTags.filter((tag) => tag !== tagToRemove)
    );
  };

  const handleTagKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      e.preventDefault();
      addTag();
    }
  };

  return (
    <div>
      <div className="mb-6">
        <Link href="/">
          <Button variant="outline" className="mb-4">
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back to Dashboard
          </Button>
        </Link>
        
        <h1 className="text-3xl font-bold">Add New Dataset</h1>
        <p className="text-neutral-medium mt-1">
          Provide dataset details to start the metadata generation process
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>Dataset Information</CardTitle>
            <CardDescription>
              Enter the details about the dataset you want to process
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Form {...form}>
              <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-6">
                {validationError && (
                  <Alert variant="destructive">
                    <AlertDescription>{validationError}</AlertDescription>
                  </Alert>
                )}

                <FormField
                  control={form.control}
                  name="name"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Dataset Name*</FormLabel>
                      <FormControl>
                        <div className="flex items-center">
                          <Database className="h-4 w-4 mr-2 text-neutral-medium" />
                          <Input placeholder="E.g., COVID-19 Case Surveillance Data" {...field} />
                        </div>
                      </FormControl>
                      <FormDescription>
                        A clear, descriptive name for the dataset
                      </FormDescription>
                      <FormMessage />
                    </FormItem>
                  )}
                />

                <FormField
                  control={form.control}
                  name="description"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Description</FormLabel>
                      <FormControl>
                        <div className="flex">
                          <FileText className="h-4 w-4 mr-2 mt-2 text-neutral-medium flex-shrink-0" />
                          <Textarea 
                            placeholder="Describe the dataset content, purpose, and potential applications" 
                            className="min-h-[100px]" 
                            {...field} 
                          />
                        </div>
                      </FormControl>
                      <FormDescription>
                        A detailed description helps with searchability and metadata quality
                      </FormDescription>
                      <FormMessage />
                    </FormItem>
                  )}
                />

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <FormField
                    control={form.control}
                    name="url"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Dataset URL*</FormLabel>
                        <FormControl>
                          <div className="flex items-center">
                            <LinkIcon className="h-4 w-4 mr-2 text-neutral-medium" />
                            <Input placeholder="https://example.com/dataset" {...field} />
                          </div>
                        </FormControl>
                        <FormDescription>
                          Direct link to the dataset file or repository
                        </FormDescription>
                        <FormMessage />
                      </FormItem>
                    )}
                  />

                  <FormField
                    control={form.control}
                    name="source"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Source</FormLabel>
                        <FormControl>
                          <Input placeholder="E.g., Kaggle, CDC.gov" {...field} />
                        </FormControl>
                        <FormDescription>
                          The organization or website providing the dataset
                        </FormDescription>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <FormField
                    control={form.control}
                    name="format"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Format</FormLabel>
                        <FormControl>
                          <Input placeholder="E.g., CSV, JSON, XML" {...field} />
                        </FormControl>
                        <FormDescription>
                          The file format of the dataset
                        </FormDescription>
                        <FormMessage />
                      </FormItem>
                    )}
                  />

                  <FormField
                    control={form.control}
                    name="size"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Size</FormLabel>
                        <FormControl>
                          <div className="flex items-center">
                            <HardDrive className="h-4 w-4 mr-2 text-neutral-medium" />
                            <Input placeholder="E.g., 2.5 GB, 500 MB" {...field} />
                          </div>
                        </FormControl>
                        <FormDescription>
                          Approximate size of the dataset
                        </FormDescription>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                </div>

                <FormField
                  control={form.control}
                  name="tags"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Tags</FormLabel>
                      <FormDescription>
                        Add relevant keywords to improve discoverability
                      </FormDescription>
                      
                      <div className="flex flex-wrap gap-2 mb-3">
                        {form.watch("tags")?.map((tag) => (
                          <div
                            key={tag}
                            className="flex items-center bg-primary bg-opacity-10 text-primary text-sm px-3 py-1 rounded-full"
                          >
                            {tag}
                            <button
                              type="button"
                              onClick={() => removeTag(tag)}
                              className="ml-1 focus:outline-none"
                            >
                              <X className="h-3 w-3" />
                            </button>
                          </div>
                        ))}
                      </div>
                      
                      <div className="flex items-center">
                        <Tag className="h-4 w-4 mr-2 text-neutral-medium" />
                        <Input
                          placeholder="Add a tag and press Enter"
                          value={tagInput}
                          onChange={(e) => setTagInput(e.target.value)}
                          onKeyDown={handleTagKeyDown}
                          className="flex-grow"
                        />
                        <Button
                          type="button"
                          variant="outline"
                          size="sm"
                          onClick={addTag}
                          className="ml-2"
                        >
                          <Plus className="h-4 w-4" />
                        </Button>
                      </div>
                      <FormMessage />
                    </FormItem>
                  )}
                />

                <Separator />

                <div className="flex justify-end space-x-4">
                  <Button variant="outline" type="button" onClick={() => form.reset()}>
                    Reset
                  </Button>
                  <Button 
                    type="submit" 
                    className="bg-primary hover:bg-primary-dark text-white"
                    disabled={addDatasetMutation.isPending}
                  >
                    {addDatasetMutation.isPending ? (
                      <>
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        Processing...
                      </>
                    ) : (
                      <>
                        <Plus className="mr-2 h-4 w-4" />
                        Add Dataset
                      </>
                    )}
                  </Button>
                </div>
              </form>
            </Form>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Processing Information</CardTitle>
            <CardDescription>
              What happens when you add a dataset
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <h3 className="text-base font-medium flex items-center">
                <span className="flex items-center justify-center w-6 h-6 rounded-full bg-primary text-white text-sm mr-2">1</span>
                Dataset Download
              </h3>
              <p className="text-neutral-medium text-sm mt-1 ml-8">
                The system will download the dataset from the provided URL
              </p>
            </div>
            
            <div>
              <h3 className="text-base font-medium flex items-center">
                <span className="flex items-center justify-center w-6 h-6 rounded-full bg-primary text-white text-sm mr-2">2</span>
                Structure Analysis
              </h3>
              <p className="text-neutral-medium text-sm mt-1 ml-8">
                The dataset structure will be analyzed for format, schema, and quality
              </p>
            </div>
            
            <div>
              <h3 className="text-base font-medium flex items-center">
                <span className="flex items-center justify-center w-6 h-6 rounded-full bg-primary text-white text-sm mr-2">3</span>
                Metadata Generation
              </h3>
              <p className="text-neutral-medium text-sm mt-1 ml-8">
                Structured metadata will be generated according to schema.org standards
              </p>
            </div>
            
            <div>
              <h3 className="text-base font-medium flex items-center">
                <span className="flex items-center justify-center w-6 h-6 rounded-full bg-primary text-white text-sm mr-2">4</span>
                FAIR Assessment
              </h3>
              <p className="text-neutral-medium text-sm mt-1 ml-8">
                The dataset will be evaluated for compliance with FAIR principles
              </p>
            </div>
            
            <div>
              <h3 className="text-base font-medium flex items-center">
                <span className="flex items-center justify-center w-6 h-6 rounded-full bg-primary text-white text-sm mr-2">5</span>
                Quality Scoring
              </h3>
              <p className="text-neutral-medium text-sm mt-1 ml-8">
                Overall quality scores and recommendations will be provided
              </p>
            </div>

            <Separator />
            
            <div className="bg-blue-50 p-4 rounded-md">
              <h4 className="font-medium text-blue-600">Processing Time</h4>
              <p className="text-sm text-blue-600 mt-1">
                Processing time varies based on dataset size and complexity. You'll be notified when processing is complete.
              </p>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
